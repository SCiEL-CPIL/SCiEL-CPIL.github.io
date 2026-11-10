"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # 1. Generate all words as integers
    total_words = q ** n
    q_powers = [q ** i for i in range(n)]
    
    def word_to_int(w):
        v = 0
        for i in range(n):
            v = v * q + w[i]
        return v

    def int_to_word(v):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = v % q
            v //= q
        return w

    # Generate all possible words
    all_words = list(range(total_words))
    
    # Precompute neighbors for each word (words within Hamming distance <= radius)
    # neighbors[i] is a list of ints
    neighbors = [[] for _ in range(total_words)]
    for i in all_words:
        w = int_to_word(i)
        neighbors[i].append(i) # distance 0
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    w[pos] = new_val
                    neighbors[i].append(word_to_int(w))
                w[pos] = orig
        # Remove duplicates just in case (though logic above prevents it for radius 1)
        # neighbors[i] = list(set(neighbors[i])) 
        
    # 2. Initialize with Direct Sum Code (81 words for n=6, q=3)
    # Tetracode for first 4 coords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code = []
    if n >= 4:
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                init_code.append(word_to_int(h + list(tail)))
    else:
        # Fallback for small n
        init_code = all_words[:]

    # Ensure distinct and valid
    init_code = list(set(init_code))
    
    # 3. Setup State
    in_code = np.zeros(total_words, dtype=bool)
    coverage_count = np.zeros(total_words, dtype=np.int32)
    code_set = set(init_code)
    
    for c in init_code:
        in_code[c] = True
        for nb in neighbors[c]:
            coverage_count[nb] += 1
            
    # 4. Local Search: Removal and Swap
    # We want to minimize len(code_set)
    
    def try_remove():
        """Try to remove any codeword that doesn't break coverage. Returns True if removed."""
        removed_any = False
        # Randomize order to avoid getting stuck in specific removal sequences
        candidates = [c for c in code_set if in_code[c]]
        random.shuffle(candidates)
        
        for c in candidates:
            if not in_code[c]:
                continue
            # Check if c is essential (covers any word uniquely)
            essential = False
            for nb in neighbors[c]:
                if coverage_count[nb] == 1:
                    essential = True
                    break
            if not essential:
                # Safe to remove
                in_code[c] = False
                code_set.remove(c)
                for nb in neighbors[c]:
                    coverage_count[nb] -= 1
                removed_any = True
        return removed_any

    def try_swap():
        """Try to swap a codeword to enable further reduction or just escape local minimum.
        Strategy: Pick a random c_out. Find c_in that covers the uniquely covered words of c_out.
        If found, perform swap. Then try to remove more."""
        if not code_set:
            return False
        
        candidates = list(code_set)
        random.shuffle(candidates)
        
        for c_out in candidates:
            if not in_code[c_out]:
                continue
            
            # Identify words uniquely covered by c_out
            unique_words = []
            for nb in neighbors[c_out]:
                if coverage_count[nb] == 1:
                    unique_words.append(nb)
            
            # If no unique words, try_remove would have caught it. 
            # So we only swap if there are unique words.
            if not unique_words:
                continue
                
            # Find a candidate c_in that covers all unique_words
            # We can restrict c_in to words not in code_set
            # To speed up, we can just iterate all words
            best_c_in = -1
            # Optimization: A valid c_in must cover all words in unique_words.
            # We can check candidates.
            # For n=6, total_words=729, this is fast.
            
            # Sort candidates by how many unique_words they cover? 
            # Or just find the first one that covers all?
            # Let's find one that covers all.
            found = False
            # Shuffle all words to pick randomly if multiple exist
            all_candidates = [w for w in all_words if not in_code[w]]
            random.shuffle(all_candidates)
            
            for c_in in all_candidates:
                # Check if c_in covers all unique_words
                valid = True
                # Optimization: Check if unique_words is subset of neighbors[c_in]
                # Since unique_words is small, we can check if each u is in neighbors[c_in]
                # Precompute a set for neighbors[c_in]? No, neighbors is list.
                # Let's just iterate unique_words.
                for u in unique_words:
                    # Check if u is in neighbors[c_in]
                    # Linear search in neighbors[c_in] is O(n*q). 
                    # 729 candidates * 10 unique * 20 neighbors = 145k ops. Acceptable.
                    if u not in neighbors[c_in]:
                        valid = False
                        break
                if valid:
                    best_c_in = c_in
                    found = True
                    break
            
            if found:
                # Perform Swap
                # Remove c_out
                in_code[c_out] = False
                code_set.remove(c_out)
                for nb in neighbors[c_out]:
                    coverage_count[nb] -= 1
                
                # Add c_in
                in_code[best_c_in] = True
                code_set.add(best_c_in)
                for nb in neighbors[best_c_in]:
                    coverage_count[nb] += 1
                
                return True
                
        return False

    # Main Loop
    while time.time() < deadline:
        # Phase 1: Greedy Removal
        while try_remove():
            pass
        
        # Phase 2: Try Swaps to escape local minima
        # We might need multiple swaps to get out of a "valley"
        swapped = False
        for _ in range(10): # Try a few swaps per iteration
            if try_swap():
                swapped = True
            else:
                break
        
        if not swapped:
            # If no swap worked, we are in a local minimum.
            # We could restart with a different initialization or random perturbation.
            # For now, just break if we've been stuck for a while? 
            # Or try a random "kick": remove a random word, add a random word (even if invalid temporarily?)
            # No, must maintain validity.
            
            # Alternative: If stuck, try to add a word that allows removal of 2 words?
            # This is complex. Let's just rely on the randomization in try_swap.
            # If try_swap fails 10 times in a row, we might be stuck in a very deep local min.
            # Let's try a "2-for-1" swap: Remove 2, Add 1.
            # This is expensive. 
            # Given time limit, let's just break if we can't improve size.
            # Actually, try_swap attempts to swap 1-for-1. 
            # If 1-for-1 doesn't reduce size (it keeps size same), it doesn't help minimization directly, 
            # UNLESS it enables subsequent removals.
            # So we MUST loop back to try_remove after a swap.
            
            # If no swap was performed, we are stuck.
            # Let's try a random perturbation: Pick a random codeword, remove it, 
            # and add a random codeword that covers the gap, even if it doesn't lead to immediate removal.
            # This is effectively what try_swap does.
            
            # If we are truly stuck, we can break and return the best found so far.
            # But we might have time for more random restarts of the local search?
            # The state is already optimized for the current set.
            # Let's just break to avoid wasting time.
            break

    # Final Code
    final_code = [int_to_word(c) for c in code_set]
    
    return final_code
# EVOLVE-BLOCK-END
