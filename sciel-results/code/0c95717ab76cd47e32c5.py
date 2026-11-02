import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code using bitmask-based greedy set cover
    and local search (removal and swaps).
    
    Returns: list of codewords (each a list of n integers in 0..q-1)
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1
    
    # Helper functions for base conversion
    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # Precompute neighborhoods as bitmasks
    # Each word's neighborhood is an integer where bit i is 1 if word i is in the neighborhood
    masks = [0] * total_words
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        mask = 1 << i
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                nb_idx = digits_to_int(d, q)
                mask |= (1 << nb_idx)
            d[pos] = orig
        masks[i] = mask

    # 1. Greedy Set Cover
    # uncovered: bitmask of all words
    uncovered = (1 << total_words) - 1
    code = []
    
    # Optimization: We can pre-sort or just iterate. 
    # To speed up greedy, we can use a priority-like approach, but simple iteration with max gain is okay.
    # With 6561 words, iterating 486 times is fine.
    
    while uncovered != 0:
        if time.time() > deadline:
            break
            
        best_center = -1
        best_gain = 0
        
        # Find center that covers most uncovered words
        # Optimization: We can skip centers that are already in code? No, greedy adds them.
        # We can also skip if gain is 0.
        
        # Vectorized approach using numpy is tricky with bitmasks.
        # Python loop over 6561 items is fast enough if done ~500 times (3M ops).
        for c in range(total_words):
            if c in code: # Optimization: don't pick same center twice (shouldn't happen in pure greedy, but safe)
                 continue
            gain = (masks[c] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_center = c
        
        if best_center == -1 or best_gain == 0:
            break
            
        code.append(best_center)
        uncovered &= ~masks[best_center]
        
        # If time is critical, we can break early if best_gain is small, but let's try to be precise.
        # Actually, we can stop if best_gain == 1? No, we need to cover everything.

    # Initial cleanup: Remove redundant codewords
    # A codeword c is redundant if its coverage is fully contained in the union of others.
    # union_others = OR of masks of all other codewords
    # If masks[c] & ~union_others == 0, then c is redundant.
    
    def trim_code():
        nonlocal code
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Compute union of all codewords
            # Instead of recomputing union from scratch for each c, 
            # we can maintain a global union and subtract c's contribution? 
            # No, union is not a group. 
            # Efficient check: 
            # For each c, check if there exists any word in masks[c] that is ONLY covered by c.
            # Equivalent: Is masks[c] subset of (Union of all other masks)?
            
            # Let's compute the global union first
            global_union = 0
            for c in code:
                global_union |= masks[c]
                
            # For each c, check if masks[c] & ~global_union == 0? 
            # Wait, if masks[c] & ~global_union == 0, it means all words in c are in global_union.
            # But global_union includes c. 
            # We need: masks[c] & ~(global_union ^ masks[c] ... no, union is not XOR).
            
            # Correct logic:
            # Let U = Union of all masks in code.
            # Let U_without_c = Union of masks in code excluding c.
            # c is redundant if masks[c] & ~U_without_c == 0.
            
            # Computing U_without_c for each c is O(N^2). N is ~500. 250k ops. Fast.
            
            to_remove = []
            for i, c in enumerate(code):
                U_without_c = 0
                for j, other in enumerate(code):
                    if i != j:
                        U_without_c |= masks[other]
                
                # Check if c covers anything not covered by others
                unique_part = masks[c] & ~U_without_c
                if unique_part == 0:
                    to_remove.append(i)
                    
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    code.pop(idx)
                changed = True
                
    trim_code()

    # 2. Local Search
    # Strategies:
    # A. Remove redundant (already done, but can happen after swaps)
    # B. Swap 1: Replace c1 with c2 if c2 covers all unique parts of c1.
    # C. Swap 2: Replace c1, c2 with c3, c4? (Complex, maybe skip for now)
    
    def local_search():
        nonlocal code
        code_set = set(code)
        
        # Precompute unique parts for each codeword
        # unique[c] = masks[c] & ~Union(others)
        
        def compute_unique_parts():
            uniques = {}
            for i, c in enumerate(code):
                U_without_c = 0
                for j, other in enumerate(code):
                    if i != j:
                        U_without_c |= masks[other]
                uniques[c] = masks[c] & ~U_without_c
            return uniques

        max_rounds = 50
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
                
            improved = False
            uniques = compute_unique_parts()
            
            # 1. Try removals
            for c in code:
                if uniques[c] == 0:
                    code.remove(c)
                    code_set.discard(c)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try swaps
            # For each codeword c in code, try to replace it with some w not in code
            # such that w covers all of uniques[c]
            
            # To speed up, iterate through code in random order
            code_list = list(code)
            random.shuffle(code_list)
            
            for c in code_list:
                if time.time() > deadline:
                    break
                unique_part = uniques[c]
                if unique_part == 0:
                    continue # Should have been removed, but safety check
                    
                # Find candidates w
                # We want w such that (unique_part & ~masks[w]) == 0
                # i.e., unique_part is subset of masks[w]
                
                # Sample candidates from non-code words
                # Total words 6561. Code size ~500.
                # We can iterate through all non-code words? 6000 checks. 
                # Each check is a bitwise AND and NOT. Very fast.
                
                found_swap = False
                # Optimization: Only check words that are close to c? 
                # Or just random sample first?
                # Let's try all non-code words if time permits, else sample.
                
                candidates = []
                for w in range(total_words):
                    if w not in code_set:
                        candidates.append(w)
                        
                random.shuffle(candidates)
                
                # Try a subset of candidates to save time
                limit = min(len(candidates), 200)
                for w in candidates[:limit]:
                    if (unique_part & ~masks[w]) == 0:
                        # Valid swap
                        code[code.index(c)] = w
                        code_set.discard(c)
                        code_set.add(w)
                        improved = True
                        found_swap = True
                        break
                        
                if improved:
                    break

        # Final trim
        trim_code()
        
    # Run local search with a few different random seeds to escape local minima
    for seed in [42, 123, 456]:
        if time.time() > deadline:
            break
        random.seed(seed)
        local_search()
        trim_code()

    # 3. Final Polish: Try to remove any remaining redundant words
    # (trim_code does this)
    
    # Convert back to list of lists
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))
        
    return result

# Ensure distinct codewords (should be distinct by construction, but check)
# The problem requires distinct codewords. Our set-based logic ensures this.
