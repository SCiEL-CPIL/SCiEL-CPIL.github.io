import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 58.0  # Safety margin
    
    def elapsed():
        return time.time() - start_time

    small_nums = list(range(2, 129))
    large_nums = list(range(129, 257))
    n_small = len(small_nums)
    n_large = len(large_nums)
    
    small_weights = [1.0 / x for x in small_nums]
    large_weights = [1.0 / x for x in large_nums]
    
    # Precompute small-small conflict mask
    # conflict[i] has bit j set if lcm(small[i], small[j]) <= 256
    small_conflict = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        for j in range(i + 1, n_small):
            b = small_nums[j]
            l = a * b // gcd(a, b)
            if l <= 256:
                small_conflict[i] |= (1 << j)
                small_conflict[j] |= (1 << i)

    # Precompute small-large blocking mask
    # blocks[i] has bit j set if lcm(small[i], large[j]) <= 256
    small_blocks_large = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        mask = 0
        for j in range(n_large):
            b = large_nums[j]
            l = a * b // gcd(a, b)
            if l <= 256:
                mask |= (1 << j)
        small_blocks_large[i] = mask

    def calc_cost(blocked_mask):
        c = 0.0
        m = blocked_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            c += large_weights[idx]
            m ^= lsb
        return c

    def calc_gain(mask):
        g = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            g += small_weights[idx]
            m ^= lsb
        return g

    def get_score(mask):
        blocked = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            blocked |= small_blocks_large[idx]
            m ^= lsb
        return calc_gain(mask) - calc_cost(blocked)

    best_score = -1.0
    best_mask = 0

    # Multiple restarts
    for restart in range(50):
        if elapsed() > DEADLINE - 2.0:
            break
            
        # Randomized Greedy Initialization
        mask = 0
        order = list(range(n_small))
        random.shuffle(order)
        for i in order:
            if not (small_conflict[i] & mask):
                # Check if adding improves score? 
                # For init, we just build a valid set. 
                # To be smarter, we could check marginal gain.
                # But simple valid set is a good start for local search.
                mask |= (1 << i)
                
        current_score = get_score(mask)
        if current_score > best_score:
            best_score = current_score
            best_mask = mask

        # Local Search
        improved = True
        while improved and elapsed() < DEADLINE - 1.0:
            improved = False
            best_move_score = current_score
            best_new_mask = mask

            # 1-Add
            for i in range(n_small):
                if mask & (1 << i):
                    continue
                if small_conflict[i] & mask:
                    continue
                new_mask = mask | (1 << i)
                s = get_score(new_mask)
                if s > best_move_score:
                    best_move_score = s
                    best_new_mask = new_mask

            # 1-Remove
            m = mask
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                new_mask = mask & ~(1 << i)
                s = get_score(new_mask)
                if s > best_move_score:
                    best_move_score = s
                    best_new_mask = new_mask
                m ^= lsb

            # 1-Swap
            selected = []
            m = mask
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                selected.append(i)
                m ^= lsb

            for i in selected:
                mask_wo_i = mask & ~(1 << i)
                for j in range(n_small):
                    if mask_wo_i & (1 << j):
                        continue
                    if small_conflict[j] & mask_wo_i:
                        continue
                    new_mask = mask_wo_i | (1 << j)
                    s = get_score(new_mask)
                    if s > best_move_score:
                        best_move_score = s
                        best_new_mask = new_mask

            if best_new_mask != mask:
                mask = best_new_mask
                current_score = best_move_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
                improved = True
            else:
                # Stuck. Try 2-Swap if time permits
                if elapsed() < DEADLINE - 3.0 and len(selected) >= 2:
                    # Try a limited number of random 2-swaps
                    # Or deterministic if small enough
                    # Let's try all 2-swaps if |selected| is small, else random
                    pairs_to_check = []
                    if len(selected) < 15:
                        for idx1 in range(len(selected)):
                            for idx2 in range(idx1 + 1, len(selected)):
                                pairs_to_check.append((selected[idx1], selected[idx2]))
                    else:
                        # Random sample of pairs
                        for _ in range(50):
                            i, j = random.sample(selected, 2)
                            pairs_to_check.append((i, j))
                            
                    found_improvement = False
                    for i, j in pairs_to_check:
                        if elapsed() > DEADLINE - 2.0:
                            break
                        mask_wo_ij = mask & ~(1 << i) & ~(1 << j)
                        # Find candidates to add
                        # Must be compatible with mask_wo_ij and each other
                        candidates = []
                        for k in range(n_small):
                            if mask_wo_ij & (1 << k):
                                continue
                            if small_conflict[k] & mask_wo_ij:
                                continue
                            candidates.append(k)
                            
                        # Check pairs in candidates
                        # This is O(C^2). C can be up to 127.
                        # 127^2 = 16k. Too slow for many i,j pairs.
                        # Optimization: Pre-filter candidates?
                        # Just check first few or random?
                        # Let's just check if there's ANY improvement in 1-swap from mask_wo_ij.
                        # We already did 1-swap on mask. 
                        # 2-swap is remove 2, add 2.
                        # If we remove i,j, we might unblock large numbers, allowing us to add k,l that were previously blocked or conflicting.
                        
                        # Efficient 2-swap:
                        # Iterate k in candidates
                        # For each k, iterate l in candidates > k
                        # Check conflict k,l
                        # Check conflict k,l with mask_wo_ij (already checked individually)
                        
                        # To keep it fast, limit candidates checked
                        if len(candidates) > 20:
                            candidates = random.sample(candidates, 20)
                            
                        for idx_k in range(len(candidates)):
                            k = candidates[idx_k]
                            for idx_l in range(idx_k + 1, len(candidates)):
                                l = candidates[idx_l]
                                if small_conflict[k] & (1 << l):
                                    continue
                                if small_conflict[l] & (1 << k):
                                    continue
                                new_mask = mask_wo_ij | (1 << k) | (1 << l)
                                s = get_score(new_mask)
                                if s > best_move_score:
                                    best_move_score = s
                                    best_new_mask = new_mask
                                    found_improvement = True
                                    break
                            if found_improvement:
                                break
                        if found_improvement:
                            break

                    if found_improvement:
                        mask = best_new_mask
                        current_score = best_move_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = mask
                        improved = True
                    else:
                        improved = False
                else:
                    improved = False

    # Construct result
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(small_nums[idx])
        m ^= lsb

    blocked = 0
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        blocked |= small_blocks_large[idx]
        m ^= lsb

    for j in range(n_large):
        if not (blocked & (1 << j)):
            result.append(large_nums[j])

    if not result:
        result = [129]
        
    return result
