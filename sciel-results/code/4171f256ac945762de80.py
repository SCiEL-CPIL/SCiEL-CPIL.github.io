import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # Specific to q=3, n=6, radius=1
    # General fallback for other params if needed, but task is specific.
    if q != 3 or n != 6 or radius != 1:
        # Fallback to simple greedy or tetracode
        # Tetracode sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = 3 ** n
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin for 120s limit

    # Precompute neighbors using NumPy for speed
    # neighbors[i] contains indices of all words within distance 1 of i
    neighbors = np.empty((total_words, 13), dtype=np.int32)
    powers = [3 ** p for p in range(n)]
    
    for i in range(total_words):
        # Decode word
        d = [(i // powers[p]) % 3 for p in range(n)]
        neighbors[i, 0] = i
        k = 1
        for p in range(n):
            orig = d[p]
            for alt in range(3):
                if alt == orig:
                    continue
                # Calculate new index
                new_i = i + (alt - orig) * powers[p]
                neighbors[i, k] = new_i
                k += 1
        # Ensure we filled exactly 13 slots (1 self + 2 per position * 6 positions)
        if k != 13:
             # This shouldn't happen for q=3, n=6, radius=1
             pass

    # Helper to convert index to list
    def idx_to_list(idx):
        return [(idx // powers[p]) % 3 for p in range(n)]

    # Initial Solution: Tetracode Sum
    # This is a known good starting point (size 72)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            idx = 0
            for w in word:
                idx = idx * 3 + w
            initial_code_indices.append(idx)
            
    # Remove duplicates just in case
    initial_code_indices = list(set(initial_code_indices))
    
    # Initialize state
    # We maintain a list of codeword indices
    current_code = list(initial_code_indices)
    # coverage_counts[w] = number of codewords in current_code that cover w
    coverage_counts = np.zeros(total_words, dtype=np.int32)
    
    for c in current_code:
        coverage_counts[neighbors[c]] += 1
        
    # Track best solution found
    best_code = list(current_code)
    best_size = len(current_code)
    
    # Helper to check if code is valid (all covered)
    def is_valid(counts):
        return np.all(counts > 0)
        
    # Helper to try to remove redundant words
    def greedy_remove():
        nonlocal current_code, coverage_counts, best_code, best_size
        improved = True
        while improved:
            improved = False
            # Find redundant words: min coverage of neighbors >= 2
            # We check one by one to avoid O(N^2) in the worst case if we recompute every time
            # But we can vectorize the check for all current codewords
            if not current_code:
                break
                
            # Vectorized check: for each c in current_code, is min(coverage_counts[neighbors[c]]) >= 2?
            # This is expensive if done in a loop. 
            # Optimization: Only check if we suspect redundancy. 
            # However, for N~70, a loop of 70 * (13 checks) is very fast in Python if we use numpy for the min.
            
            to_remove = []
            for idx, c in enumerate(current_code):
                if coverage_counts[neighbors[c]].min() >= 2:
                    to_remove.append(idx)
                    
            if to_remove:
                # Remove one redundant word (prefer the one with highest slack? or random?)
                # Let's pick the first one for determinism in polish, random in search
                if random.random() < 0.5 and len(to_remove) > 1:
                    idx_to_remove = random.choice(to_remove)
                else:
                    idx_to_remove = to_remove[0]
                    
                c_to_remove = current_code[idx_to_remove]
                current_code.pop(idx_to_remove)
                coverage_counts[neighbors[c_to_remove]] -= 1
                improved = True
                
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                    
    # Main Search Loop
    # We alternate between Greedy Remove and Swap moves
    
    steps = 0
    last_improvement = time.time()
    no_improvement_counter = 0
    max_no_improvement = 500 # Reset or adjust strategy if stuck
    
    while time.time() < deadline - 5: # Leave 5s for final polish
        steps += 1
        
        # 1. Try to remove redundant words
        greedy_remove()
        
        if time.time() > deadline - 5:
            break
            
        # 2. If no redundant words, try a Swap
        # Check if any word has coverage 1 (critical)
        # Actually, we just pick a random codeword to potentially swap out
        if len(current_code) > 0:
            idx_to_swap = random.randint(0, len(current_code) - 1)
            c_out = current_code[idx_to_swap]
            
            # Find critical words: neighbors of c_out with coverage == 1
            # If we remove c_out, these words will have coverage 0 (invalid)
            # So c_in MUST cover all critical words.
            crit_mask = (coverage_counts[neighbors[c_out]] == 1)
            critical_words = neighbors[c_out][crit_mask]
            
            if len(critical_words) == 0:
                # c_out is redundant? Should have been caught by greedy_remove.
                # But maybe coverage changed?
                # If no critical words, we can just remove it.
                current_code.pop(idx_to_swap)
                coverage_counts[neighbors[c_out]] -= 1
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                continue
            
            # Find candidates for c_in
            # c_in must be in the intersection of neighbors of all critical_words
            # Start with neighbors of the first critical word
            w0 = int(critical_words[0])
            candidates = neighbors[w0]
            
            valid_candidates = []
            # Filter candidates
            # A candidate c is valid if:
            # 1. c is not c_out
            # 2. c covers all critical_words (i.e., critical_words is subset of neighbors[c])
            # 3. Ideally, c is not already in current_code (to avoid duplicates, though duplicates are harmless if we treat as multiset, but we want distinct)
            
            for cand in candidates:
                cand_int = int(cand)
                if cand_int == c_out:
                    continue
                    
                # Check if cand covers all critical words
                # critical_words is an array of indices
                # We need to check if all w in critical_words are in neighbors[cand_int]
                # Efficient check: 
                # neighbors[cand_int] is an array of 13 ints.
                # We can check intersection size.
                # Since critical_words is small (usually 1 or 2), we can just loop
                is_valid = True
                for w in critical_words:
                    if w not in neighbors[cand_int]:
                        is_valid = False
                        break
                if is_valid:
                    # Optional: Prefer candidates not already in code
                    # Checking membership in a list is O(N). 
                    # For N=70, this is fine.
                    # But we can just allow duplicates in the search space and clean up later?
                    # No, distinct codewords are required. 
                    # If cand is already in code, swapping it for c_out is just a removal of c_out (if cand==c_out, handled) 
                    # or a no-op if we add cand and remove c_out, but cand is already there.
                    # If cand is already in code, adding it increases coverage, removing c_out decreases.
                    # Net effect: coverage of c_out's neighbors decreases by 1 (except where covered by cand), 
                    # coverage of cand's neighbors increases by 1.
                    # This is a valid move.
                    valid_candidates.append(cand_int)
            
            if valid_candidates:
                # Pick a candidate
                # Heuristic: Pick one that maximizes redundancy? 
                # Or just random.
                c_in = random.choice(valid_candidates)
                
                # Perform swap
                # Update coverage
                coverage_counts[neighbors[c_out]] -= 1
                coverage_counts[neighbors[c_in]] += 1
                
                # Update code list
                current_code[idx_to_swap] = c_in
                
                # Check for new redundancies created by the swap
                # The swap might have made some words redundant.
                # We handle this in the next iteration's greedy_remove.
                
                # Check if size improved (it didn't, it's a swap)
                # But it might enable future removals.
                
                # If this swap led to a state where we can remove more, we'll catch it next loop.
                
        else:
            # Code is empty? Should not happen for this problem size.
            break
            
        # Track improvement
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)
            last_improvement = time.time()
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
            
        # If stuck for a long time, try a more aggressive reset or random restart?
        # For now, the swap mechanism should provide enough diversity.
        
    # Final Polish: Ensure we have the absolute smallest code from the best trajectory
    # We take best_code and try to reduce it further
    # Note: best_code is a snapshot. We should verify it's valid and try to reduce it.
    
    # Reconstruct state from best_code
    final_code = list(best_code)
    final_counts = np.zeros(total_words, dtype=np.int32)
    for c in final_code:
        final_counts[neighbors[c]] += 1
        
    # Verify validity
    if not np.all(final_counts > 0):
        # If best_code is invalid (should not happen), fall back to current_code
        final_code = list(current_code)
        final_counts = coverage_counts.copy()
        
    # Aggressive greedy removal on the best code
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        to_remove = []
        for idx, c in enumerate(final_code):
            if final_counts[neighbors[c]].min() >= 2:
                to_remove.append(idx)
                
        if to_remove:
            # Remove all redundant words in this pass? 
            # Removing multiple at once is safe if they are all redundant *in the current state*.
            # But removing one might make another non-redundant? No, removing a codeword only decreases counts.
            # So if min count >= 2, it stays >= 1 after removing another codeword.
            # So we can remove all identified redundant words in one batch.
            for idx in sorted(to_remove, reverse=True):
                c = final_code[idx]
                final_code.pop(idx)
                final_counts[neighbors[c]] -= 1
            improved = True
            
    # Final check
    if not np.all(final_counts > 0):
        # Fallback to initial tetracode if something went wrong
        return [idx_to_list(idx) for idx in initial_code_indices]
        
    return [idx_to_list(idx) for idx in final_code]
