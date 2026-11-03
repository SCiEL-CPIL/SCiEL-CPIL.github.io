import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    """
    Construct a ternary covering code K_3(6,1).
    Returns a list of codewords (lists of integers).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Fallback for non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        if n <= 4:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    total_words = 3 ** n  # 729
    
    # --- 1. Precompute Data Structures ---
    
    # Helper: Convert index to word tuple
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[n - 1 - i] = temp % 3
            temp //= 3
        return tuple(w)
    
    # Helper: Convert word tuple to index
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * 3 + val
        return idx

    # Precompute spheres as bitmasks
    # sphere_masks[i] is an integer with bits set for all words in sphere of i
    # coverer_masks[w] is an integer with bits set for all codewords covering w
    sphere_masks = [0] * total_words
    coverer_masks = [0] * total_words
    
    # We also need the list of indices in each sphere for iteration in some cases
    sphere_lists = [None] * total_words
    
    for idx in range(total_words):
        w = list(idx_to_word(idx))
        mask = 1 << idx
        lst = [idx]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                new_idx = word_to_idx(w)
                mask |= (1 << new_idx)
                lst.append(new_idx)
                w[pos] = orig_val
        
        sphere_masks[idx] = mask
        sphere_lists[idx] = lst
        
        # Update coverer masks
        for w_idx in lst:
            coverer_masks[w_idx] |= (1 << idx)

    # Helper to pick a random index from a bitmask
    def pick_random_bit(mask):
        if mask == 0:
            return -1
        # Extract all set bits
        # Since N=729, this is at most 729 iterations, usually much less
        indices = []
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            indices.append(idx)
            m ^= lsb
        return random.choice(indices)

    # --- 2. Initial Code Construction ---
    # Tetracode seed
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(h + list(tail))
    
    initial_indices = [word_to_idx(cw) for cw in initial_codewords]
    
    # --- 3. Simulated Annealing ---
    
    def compute_counts(code_set):
        counts = np.zeros(total_words, dtype=np.int32)
        for idx in code_set:
            for w in sphere_lists[idx]:
                counts[w] += 1
        return counts

    def compute_count_1_mask(counts):
        mask = 0
        for i in range(total_words):
            if counts[i] == 1:
                mask |= (1 << i)
        return mask

    def is_valid(counts):
        return np.all(counts >= 1)

    def get_uncovered_mask(counts):
        mask = 0
        for i in range(total_words):
            if counts[i] == 0:
                mask |= (1 << i)
        return mask

    def can_remove(c, current_code_mask, count_1_mask):
        # c is removable if no word in its sphere has count 1
        # i.e., sphere_masks[c] & count_1_mask == 0
        return (sphere_masks[c] & count_1_mask) == 0

    def find_replacements(c_remove, current_code_mask, count_1_mask, counts):
        """
        Find all c_add such that swapping c_remove with c_add keeps validity.
        Returns a bitmask of valid c_adds.
        """
        # Words that would become uncovered if c_remove is removed
        # These are words in sphere(c_remove) with count == 1
        at_risk_mask = sphere_masks[c_remove] & count_1_mask
        
        if at_risk_mask == 0:
            # c_remove is redundant, any c_add works, but we usually don't add
            # Return mask of all non-current codewords?
            # Actually, if we just want to swap, we need a valid replacement.
            # If redundant, we can just remove it. But this function is for swaps.
            # Let's return 0 to indicate no specific replacement needed, 
            # or let the caller handle removal.
            # In our logic, if can_remove is true, we remove.
            # So this function is called when can_remove is FALSE.
            # So at_risk_mask != 0 is guaranteed if called correctly.
            return 0

        # c_add must cover all at_risk words
        # candidate_mask is intersection of coverer_masks for all at_risk words
        candidate_mask = -1  # All bits set
        
        # Iterate through set bits in at_risk_mask
        m = at_risk_mask
        while m:
            lsb = m & -m
            w_idx = lsb.bit_length() - 1
            candidate_mask &= coverer_masks[w_idx]
            m ^= lsb
            
            # Early termination if no candidates left
            if candidate_mask == 0:
                break

        # Exclude current codewords
        valid_adds_mask = candidate_mask & ~current_code_mask
        return valid_adds_mask

    # Run simulated annealing
    rng = random.Random(12345)
    
    best_code = None
    best_counts = None
    best_len = float('inf')
    
    # Multiple restarts
    max_restarts = 8
    time_per_restart = (deadline - start_time) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize
        if restart == 0:
            current_code = set(initial_indices)
        else:
            # Perturb best code
            if best_code is None:
                current_code = set(initial_indices)
            else:
                current_code = set(best_code)
                # Randomly swap some codewords
                swaps = rng.randint(3, 8)
                current_counts = compute_counts(current_code)
                current_count_1_mask = compute_count_1_mask(current_counts)
                current_code_mask = 0
                for c in current_code:
                    current_code_mask |= (1 << c)
                    
                for _ in range(swaps):
                    if len(current_code) < 2:
                        break
                    c_remove = rng.choice(list(current_code))
                    valid_adds_mask = find_replacements(c_remove, current_code_mask, current_count_1_mask, current_counts)
                    c_add = pick_random_bit(valid_adds_mask)
                    if c_add != -1:
                        current_code.remove(c_remove)
                        current_code.add(c_add)
                        # Update counts
                        for w in sphere_lists[c_remove]:
                            current_counts[w] -= 1
                        for w in sphere_lists[c_add]:
                            current_counts[w] += 1
                        # Update masks
                        current_code_mask &= ~(1 << c_remove)
                        current_code_mask |= (1 << c_add)
                        current_count_1_mask = compute_count_1_mask(current_counts)
        
        current_counts = compute_counts(current_code)
        current_count_1_mask = compute_count_1_mask(current_counts)
        current_code_mask = 0
        for c in current_code:
            current_code_mask |= (1 << c)
        
        if not is_valid(current_counts):
            # Repair: add random codewords until valid
            while not is_valid(current_counts):
                uncovered_mask = get_uncovered_mask(current_counts)
                if uncovered_mask == 0:
                    break
                w = pick_random_bit(uncovered_mask)
                candidates_mask = coverer_masks[w] & ~current_code_mask
                c_add = pick_random_bit(candidates_mask)
                if c_add == -1:
                    # All codewords covering w are already in code?
                    # Pick any new codeword
                    all_mask = (1 << total_words) - 1
                    candidates_mask = all_mask & ~current_code_mask
                    c_add = pick_random_bit(candidates_mask)
                    if c_add == -1:
                        break
                current_code.add(c_add)
                current_code_mask |= (1 << c_add)
                for w2 in sphere_lists[c_add]:
                    current_counts[w2] += 1
                current_count_1_mask = compute_count_1_mask(current_counts)
        
        # Simulated annealing
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        max_iter = 50000
        iter_count = 0
        
        cur_len = len(current_code)
        
        while time.time() < deadline - 3.0 and iter_count < max_iter:
            iter_count += 1
            
            # Try random move
            move_type = rng.random()
            
            if move_type < 0.6:
                # Try to remove a codeword
                c_remove = rng.choice(list(current_code))
                if can_remove(c_remove, current_code_mask, current_count_1_mask):
                    # Valid removal
                    current_code.remove(c_remove)
                    current_code_mask &= ~(1 << c_remove)
                    for w in sphere_lists[c_remove]:
                        current_counts[w] -= 1
                    cur_len -= 1
                    current_count_1_mask = compute_count_1_mask(current_counts)
                    if cur_len < best_len:
                        best_len = cur_len
                        best_code = current_code.copy()
                        best_counts = current_counts.copy()
                else:
                    # Try to swap
                    valid_adds_mask = find_replacements(c_remove, current_code_mask, current_count_1_mask, current_counts)
                    c_add = pick_random_bit(valid_adds_mask)
                    if c_add != -1:
                        current_code.remove(c_remove)
                        current_code.add(c_add)
                        current_code_mask &= ~(1 << c_remove)
                        current_code_mask |= (1 << c_add)
                        for w in sphere_lists[c_remove]:
                            current_counts[w] -= 1
                        for w in sphere_lists[c_add]:
                            current_counts[w] += 1
                        current_count_1_mask = compute_count_1_mask(current_counts)
                        # cur_len unchanged
                        if cur_len < best_len:
                            best_len = cur_len
                            best_code = current_code.copy()
                            best_counts = current_counts.copy()
                    else:
                        # No valid swap; try adding a random codeword (worsening)
                        if len(current_code) < 100:
                            c_add = rng.randint(0, total_words - 1)
                            if c_add not in current_code:
                                current_code.add(c_add)
                                current_code_mask |= (1 << c_add)
                                for w in sphere_lists[c_add]:
                                    current_counts[w] += 1
                                cur_len += 1
                                current_count_1_mask = compute_count_1_mask(current_counts)
            elif move_type < 0.8:
                # Try to add a codeword (worsening move, helps escape local optima)
                if len(current_code) < 100:
                    c_add = rng.randint(0, total_words - 1)
                    if c_add not in current_code:
                        current_code.add(c_add)
                        current_code_mask |= (1 << c_add)
                        for w in sphere_lists[c_add]:
                            current_counts[w] += 1
                        cur_len += 1
                        current_count_1_mask = compute_count_1_mask(current_counts)
            else:
                # Try to remove a random codeword even if invalid (will be repaired)
                c_remove = rng.choice(list(current_code))
                current_code.remove(c_remove)
                current_code_mask &= ~(1 << c_remove)
                for w in sphere_lists[c_remove]:
                    current_counts[w] -= 1
                cur_len -= 1
                current_count_1_mask = compute_count_1_mask(current_counts)
                
                # Repair
                while not is_valid(current_counts):
                    uncovered_mask = get_uncovered_mask(current_counts)
                    if uncovered_mask == 0:
                        break
                    w = pick_random_bit(uncovered_mask)
                    candidates_mask = coverer_masks[w] & ~current_code_mask
                    c_add = pick_random_bit(candidates_mask)
                    if c_add == -1:
                        all_mask = (1 << total_words) - 1
                        candidates_mask = all_mask & ~current_code_mask
                        c_add = pick_random_bit(candidates_mask)
                        if c_add == -1:
                            break
                    current_code.add(c_add)
                    current_code_mask |= (1 << c_add)
                    for w2 in sphere_lists[c_add]:
                        current_counts[w2] += 1
                    cur_len += 1
                    current_count_1_mask = compute_count_1_mask(current_counts)
                    if cur_len < best_len:
                        best_len = cur_len
                        best_code = current_code.copy()
                        best_counts = current_counts.copy()
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
        
        # After SA, try greedy reduction
        changed = True
        while changed:
            changed = False
            for c in list(current_code):
                if can_remove(c, current_code_mask, current_count_1_mask):
                    current_code.remove(c)
                    current_code_mask &= ~(1 << c)
                    for w in sphere_lists[c]:
                        current_counts[w] -= 1
                    cur_len -= 1
                    current_count_1_mask = compute_count_1_mask(current_counts)
                    changed = True
            if cur_len < best_len:
                best_len = cur_len
                best_code = current_code.copy()
                best_counts = current_counts.copy()
    
    # Final validation
    if best_code is None:
        best_code = set(initial_indices)
        best_counts = compute_counts(best_code)
    
    # Ensure validity
    if not is_valid(best_counts):
        # Fallback
        fallback = set(initial_indices)
        fb_counts = compute_counts(fallback)
        while not is_valid(fb_counts):
            uncovered_mask = get_uncovered_mask(fb_counts)
            if uncovered_mask == 0:
                break
            w = pick_random_bit(uncovered_mask)
            fb_mask = 0
            for c in fallback:
                fb_mask |= (1 << c)
            candidates_mask = coverer_masks[w] & ~fb_mask
            c_add = pick_random_bit(candidates_mask)
            if c_add == -1:
                all_mask = (1 << total_words) - 1
                candidates_mask = all_mask & ~fb_mask
                c_add = pick_random_bit(candidates_mask)
                if c_add == -1:
                    break
            fallback.add(c_add)
            for w2 in sphere_lists[c_add]:
                fb_counts[w2] += 1
        best_code = fallback
        best_counts = fb_counts
    
    return [list(idx_to_word(idx)) for idx in best_code]
