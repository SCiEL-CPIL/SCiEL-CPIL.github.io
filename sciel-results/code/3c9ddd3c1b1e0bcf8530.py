def construct():
    import time
    import random
    import numpy as np

    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N = 17
    MIN_D = 6
    SIZE = 1 << N  # 131072

    # Precompute weights for all 17-bit words
    weights = np.fromiter((i.bit_count() for i in range(SIZE)), dtype=np.int8, count=SIZE)

    def get_linear_code():
        """
        Construct the [17,8,6] linear code (256 words).
        """
        basis = []
        code = [0]
        
        for w in range(1, SIZE):
            if len(basis) == 8:
                break
            if time.time() > deadline - 50:
                break
            
            valid = True
            for c in code:
                if weights[w ^ c] < MIN_D:
                    valid = False
                    break
            
            if valid:
                basis.append(w)
                new_words = [c ^ w for c in code]
                code.extend(new_words)
        
        return code

    # Step 1: Get the 256-word linear code
    lin_code = get_linear_code()
    
    if len(lin_code) < 256:
        return lin_code

    # Step 2: Try to extend the code
    # We maintain a list of codewords `current_code`
    current_code = list(lin_code)
    current_set = set(current_code)

    # Helper to check if a word can be added
    def can_add(w, code_list, code_set):
        if w in code_set:
            return False
        for c in code_list:
            if weights[w ^ c] < MIN_D:
                return False
        return True

    # Helper to find a word that can be added
    def find_addable_word(code_list, code_set):
        # Iterate through all words
        # Optimization: We can skip words already in code
        for w in range(SIZE):
            if w in code_set:
                continue
            if can_add(w, code_list, code_set):
                return w
        return None

    # Helper to build masks for 1-for-2 swap search
    # Mask[i] is a boolean array of size SIZE, True if d(w, code[i]) <= 5
    def build_masks(code_list):
        masks = []
        for c in code_list:
            mask = np.zeros(SIZE, dtype=bool)
            # Find all w such that d(w, c) <= 5
            # w = c ^ diff, where weight(diff) <= 5
            # Precompute diffs with weight <= 5?
            # There are sum(C(17, i) for i in 0..5) = 9402 such diffs.
            # We can precompute the list of diffs once.
            pass
        return masks

    # Precompute all diffs with weight <= 5
    diffs_le_5 = []
    for d in range(SIZE):
        if weights[d] <= 5:
            diffs_le_5.append(d)
    diffs_le_5 = np.array(diffs_le_5, dtype=np.int32)

    def build_masks_fast(code_list):
        masks = []
        for c in code_list:
            mask = np.zeros(SIZE, dtype=bool)
            # w = c ^ diff
            # Since c is fixed, we just XOR c with all diffs
            # This creates an array of indices
            indices = (c ^ diffs_le_5)
            mask[indices] = True
            masks.append(mask)
        return masks

    # Main search loop
    # We will try to add words or perform swaps
    # To avoid infinite loops, we limit the number of attempts or time
    
    # First, try simple additions
    while True:
        if time.time() > deadline - 5:
            break
        w = find_addable_word(current_code, current_set)
        if w is not None:
            current_code.append(w)
            current_set.add(w)
        else:
            break

    # If we didn't reach 258, try 1-for-2 swaps
    # We need to find c in current_code, and u, v not in current_code
    # such that d(u, current_code \ {c}) >= 6, d(v, current_code \ {c}) >= 6, d(u, v) >= 6
    
    # Precompute masks for current code
    # This is expensive, so we do it once per "round" of swaps
    
    swap_attempts = 0
    max_swap_attempts = 1000 # Limit to prevent hanging if no swaps found
    
    while swap_attempts < max_swap_attempts and time.time() < deadline - 5:
        swap_attempts += 1
        
        # Rebuild masks for the current code
        # Optimization: Only rebuild if code changed? 
        # For now, rebuild every time to be safe, but it's O(256 * 9402) ~ 2.4M ops. 
        # This is fast enough in NumPy.
        
        masks = build_masks_fast(current_code)
        
        # Try each codeword as a candidate for removal
        # Shuffle indices to try different ones
        indices = list(range(len(current_code)))
        random.shuffle(indices)
        
        found_swap = False
        for i in indices:
            if time.time() > deadline - 5:
                break
                
            # Calculate forbidden region: union of masks for all j != i
            # forbidden = OR(masks[j] for j != i)
            # We can do this by OR-ing all masks, then... wait.
            # OR of all masks includes mask[i].
            # We want OR of masks[j] for j != i.
            # This is (OR of all masks) AND (NOT mask[i])? No.
            # Let U = OR(all masks).
            # We want V = OR(masks[j] for j != i).
            # Note that V is a subset of U.
            # Actually, it's easier to just OR the specific masks.
            # But that's O(N) ORs of size SIZE.
            
            # Alternative:
            # valid_words are those NOT in V.
            # w is valid if d(w, c_j) >= 6 for all j != i.
            
            # Let's compute the union of masks for j != i.
            # To speed this up, we can precompute the total union U.
            # But we need to exclude mask[i]'s contribution?
            # No, mask[i] might cover words that are also covered by other masks.
            # Simply OR-ing N-1 arrays is O(N * SIZE).
            # N=256, SIZE=131072. 256 * 131072 = 33M ops.
            # This might be slow in a Python loop if done for every i.
            
            # Better approach:
            # For a fixed i, we want to find w such that:
            # 1. w is not in current_set
            # 2. d(w, c_j) >= 6 for all j != i
            
            # Let's iterate through a subset of candidate words for u and v?
            # No, let's stick to the mask approach but optimize the OR.
            
            # We can compute the OR of all masks once: U_all.
            # Then for a specific i, the forbidden set is NOT (U_all AND NOT mask[i])? No.
            # The forbidden set is the union of masks[j] for j != i.
            # Let's just do the OR for each i, but break early if we find a swap.
            
            # Optimization: 
            # If we find a swap, we break.
            
            # Compute forbidden mask for this i
            # Start with mask 0
            # forbidden = np.zeros(SIZE, dtype=bool)
            # for j in range(len(current_code)):
            #     if j != i:
            #         forbidden |= masks[j]
            
            # This is 255 OR operations of 131072 booleans.
            # In NumPy, this is fast.
            
            forbidden = np.zeros(SIZE, dtype=bool)
            for j in range(len(current_code)):
                if j != i:
                    forbidden |= masks[j]
            
            # Find valid words: ~forbidden
            valid_indices = np.where(~forbidden)[0]
            
            # Filter out words already in current_set
            # valid_indices is an array of ints.
            # We need to check if they are in current_set.
            # Since current_set is a Python set, checking membership is O(1).
            
            candidates = []
            for w in valid_indices:
                if w not in current_set:
                    candidates.append(w)
            
            if len(candidates) < 2:
                continue
            
            # Check if any pair in candidates has distance >= 6
            # candidates is likely small.
            found_pair = False
            for u_idx in range(len(candidates)):
                u = candidates[u_idx]
                for v_idx in range(u_idx + 1, len(candidates)):
                    v = candidates[v_idx]
                    if weights[u ^ v] >= MIN_D:
                        # Found a swap!
                        # Remove current_code[i], add u, v
                        removed_c = current_code[i]
                        current_code.pop(i)
                        current_set.remove(removed_c)
                        
                        current_code.append(u)
                        current_set.add(u)
                        current_code.append(v)
                        current_set.add(v)
                        
                        found_swap = True
                        found_pair = True
                        break
                if found_pair:
                    break
            
            if found_swap:
                # Restart the outer while loop to try more swaps/adds
                break
            
        if not found_swap:
            # No swap found in this round
            break

    return current_code
