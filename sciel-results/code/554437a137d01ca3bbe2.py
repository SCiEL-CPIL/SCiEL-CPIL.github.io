"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # seconds, safety margin before 60s kill
    
    def elapsed():
        return time.time() - start_time

    N = 256
    # Split into small [2, 128] and large [129, 256]
    small_nums = list(range(2, 129))
    large_nums = list(range(129, 257))
    
    n_small = len(small_nums)
    n_large = len(large_nums)
    
    # Weights
    small_weights = [1.0 / x for x in small_nums]
    large_weights = [1.0 / x for x in large_nums]
    total_large_weight = sum(large_weights)
    
    # Precompute small-small conflict mask
    # small_conflict_mask[i] has bit j set if lcm(small_nums[i], small_nums[j]) <= 256
    small_conflict_mask = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        for j in range(i + 1, n_small):
            b = small_nums[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                small_conflict_mask[i] |= (1 << j)
                small_conflict_mask[j] |= (1 << i)
                
    # Precompute large blocking mask for each small number
    # large_block_mask[i] has bit k set if lcm(small_nums[i], large_nums[k]) <= 256
    large_block_mask = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        mask = 0
        for k in range(n_large):
            b = large_nums[k]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                mask |= (1 << k)
        large_block_mask[i] = mask

    # Precompute large weight prefix sums for fast blocked sum
    large_prefix = [0.0] * (n_large + 1)
    for k in range(n_large):
        large_prefix[k + 1] = large_prefix[k] + large_weights[k]
    
    def large_sum_from_mask(mask):
        """Sum of large_weights for indices set in mask."""
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += large_weights[idx]
            m ^= lsb
        return s
    
    def get_score(small_mask, blocked_large_mask):
        """
        Calculate total reciprocal sum for a given small_mask and the resulting blocked_large_mask.
        """
        # Sum of small weights
        s_small = 0.0
        m = small_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s_small += small_weights[idx]
            m ^= lsb
            
        # Sum of unblocked large weights
        s_blocked = large_sum_from_mask(blocked_large_mask)
        s_large = total_large_weight - s_blocked
        return s_small + s_large

    def get_blocked_large_mask(small_mask):
        blocked = 0
        m = small_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            blocked |= large_block_mask[idx]
            m ^= lsb
        return blocked

    def is_valid_small_mask(mask):
        """Check if small_mask is an independent set in the small conflict graph."""
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if (small_conflict_mask[idx] & mask) != 0:
                return False
            m ^= lsb
        return True

    best_score = 0.0
    best_small_mask = 0
    best_blocked_large_mask = 0
    
    def update_best(sm, blm, score):
        nonlocal best_score, best_small_mask, best_blocked_large_mask
        if score > best_score:
            best_score = score
            best_small_mask = sm
            best_blocked_large_mask = blm

    def local_search(initial_mask):
        # Ensure initial mask is valid
        if not is_valid_small_mask(initial_mask):
            # Remove conflicting nodes greedily
            temp_mask = 0
            m = initial_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if (small_conflict_mask[idx] & temp_mask) == 0:
                    temp_mask |= (1 << idx)
                m ^= lsb
            initial_mask = temp_mask
            
        current_mask = initial_mask
        current_blocked = get_blocked_large_mask(current_mask)
        current_score = get_score(current_mask, current_blocked)
        
        update_best(current_mask, current_blocked, current_score)
        
        improved = True
        iterations = 0
        max_iterations = 2000 # Increased safety cap
        
        while improved and elapsed() < DEADLINE and iterations < max_iterations:
            improved = False
            iterations += 1
            
            best_move_score = current_score
            best_move_type = None
            best_move_idx = None
            best_move_new_mask = current_mask
            best_move_new_blocked = current_blocked
            
            # Try Additions
            for i in range(n_small):
                if current_mask & (1 << i):
                    continue
                # Check small-small conflict
                if small_conflict_mask[i] & current_mask:
                    continue
                
                new_mask = current_mask | (1 << i)
                new_blocked = current_blocked | large_block_mask[i]
                
                new_score = get_score(new_mask, new_blocked)
                
                if new_score > best_move_score + 1e-12:
                    best_move_score = new_score
                    best_move_type = 'add'
                    best_move_idx = i
                    best_move_new_mask = new_mask
                    best_move_new_blocked = new_blocked
            
            # Try Removals
            for i in range(n_small):
                if not (current_mask & (1 << i)):
                    continue
                
                new_mask = current_mask & ~(1 << i)
                new_blocked = get_blocked_large_mask(new_mask)
                new_score = get_score(new_mask, new_blocked)
                
                if new_score > best_move_score + 1e-12:
                    best_move_score = new_score
                    best_move_type = 'remove'
                    best_move_idx = i
                    best_move_new_mask = new_mask
                    best_move_new_blocked = new_blocked
            
            # Try 1-Swaps (Remove i, Add j)
            if elapsed() < DEADLINE - 3.0:
                for i in range(n_small):
                    if not (current_mask & (1 << i)):
                        continue
                    mask_without_i = current_mask & ~(1 << i)
                    blocked_without_i = get_blocked_large_mask(mask_without_i)
                    
                    best_add_score = get_score(mask_without_i, blocked_without_i)
                    best_add_idx = -1
                    best_add_mask = mask_without_i
                    best_add_blocked = blocked_without_i
                    
                    for j in range(n_small):
                        if mask_without_i & (1 << j):
                            continue
                        if small_conflict_mask[j] & mask_without_i:
                            continue
                        
                        candidate_mask = mask_without_i | (1 << j)
                        candidate_blocked = blocked_without_i | large_block_mask[j]
                        candidate_score = get_score(candidate_mask, candidate_blocked)
                        
                        if candidate_score > best_add_score + 1e-12:
                            best_add_score = candidate_score
                            best_add_idx = j
                            best_add_mask = candidate_mask
                            best_add_blocked = candidate_blocked
                    
                    if best_add_idx != -1 and best_add_score > best_move_score + 1e-12:
                        best_move_score = best_add_score
                        best_move_type = 'swap'
                        best_move_idx = (i, best_add_idx)
                        best_move_new_mask = best_add_mask
                        best_move_new_blocked = best_add_blocked
            
            # Try 2-Swaps (Remove i, Remove k, Add j, Add l) - only if time permits
            if elapsed() < DEADLINE - 5.0:
                # Get list of elements in current mask
                in_mask = []
                m = current_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    in_mask.append(idx)
                    m ^= lsb
                
                # Try all pairs of removals
                for ii in range(len(in_mask)):
                    for kk in range(ii + 1, len(in_mask)):
                        i = in_mask[ii]
                        k = in_mask[kk]
                        
                        # Remove i and k
                        mask_without_ik = current_mask & ~(1 << i) & ~(1 << k)
                        blocked_without_ik = get_blocked_large_mask(mask_without_ik)
                        
                        # Try adding pairs j, l that don't conflict with mask_without_ik
                        for j in range(n_small):
                            if mask_without_ik & (1 << j):
                                continue
                            if small_conflict_mask[j] & mask_without_ik:
                                continue
                            
                            mask_with_j = mask_without_ik | (1 << j)
                            blocked_with_j = blocked_without_ik | large_block_mask[j]
                            
                            for l in range(j + 1, n_small):
                                if mask_with_j & (1 << l):
                                    continue
                                if small_conflict_mask[l] & mask_with_j:
                                    continue
                                # j and l must not conflict with each other
                                if (small_conflict_mask[j] & (1 << l)) or (small_conflict_mask[l] & (1 << j)):
                                    continue
                                
                                candidate_mask = mask_with_j | (1 << l)
                                candidate_blocked = blocked_with_j | large_block_mask[l]
                                candidate_score = get_score(candidate_mask, candidate_blocked)
                                
                                if candidate_score > best_move_score + 1e-12:
                                    best_move_score = candidate_score
                                    best_move_type = '2swap'
                                    best_move_idx = (i, k, j, l)
                                    best_move_new_mask = candidate_mask
                                    best_move_new_blocked = candidate_blocked

            if best_move_type:
                current_mask = best_move_new_mask
                current_blocked = best_move_new_blocked
                current_score = best_move_score
                improved = True
                update_best(current_mask, current_blocked, current_score)

        return current_mask, current_blocked, current_score

    # Strategy 1: Start with empty small set
    m, b, s = local_search(0)
    
    # Strategy 2: Greedy construction of small set
    greedy_mask = 0
    # Sort small numbers by potential gain (weight - approx cost)
    # Approx cost is sum of blocked large weights
    gains = []
    for i in range(n_small):
        blocked = large_block_mask[i]
        cost = 0.0
        mb = blocked
        while mb:
            lsb = mb & -mb
            idx = lsb.bit_length() - 1
            cost += large_weights[idx]
            mb ^= lsb
        gain = small_weights[i] - cost
        gains.append((gain, i))
    
    gains.sort(key=lambda x: -x[0])
    
    for _, i in gains:
        if elapsed() > DEADLINE - 5:
            break
        if greedy_mask & (1 << i):
            continue
        if small_conflict_mask[i] & greedy_mask:
            continue
        # Check if adding improves score
        new_mask = greedy_mask | (1 << i)
        new_blocked = get_blocked_large_mask(new_mask)
        new_score = get_score(new_mask, new_blocked)
        old_score = get_score(greedy_mask, get_blocked_large_mask(greedy_mask))
        if new_score > old_score:
            greedy_mask = new_mask
            
    m, b, s = local_search(greedy_mask)
    
    # Strategy 3: Random restarts
    random.seed(42)
    for _ in range(30):  # Reduced to leave time for 2-swap neighborhood
        if elapsed() > DEADLINE - 15:
            break
        rand_mask = 0
        # Randomly pick a few small numbers that don't conflict
        candidates = list(range(n_small))
        random.shuffle(candidates)
        for i in candidates:
            if rand_mask & (1 << i):
                continue
            if small_conflict_mask[i] & rand_mask:
                continue
            if random.random() < 0.5:
                rand_mask |= (1 << i)
                
        m, b, s = local_search(rand_mask)
    
    # Final polish: run local search on the best solution found so far
    if elapsed() < DEADLINE - 5:
        m, b, s = local_search(best_small_mask)

    # Convert best_small_mask to list of numbers
    result = []
    m = best_small_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(small_nums[idx])
        m ^= lsb
        
    # Add all large numbers that are NOT blocked
    b = best_blocked_large_mask
    for k in range(n_large):
        if not (b & (1 << k)):
            result.append(large_nums[k])
            
    result.sort()
    
    # Final validation check (should be fast)
    # Check all pairs
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            a, b = result[i], result[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= 256:
                # Invalid result, return a safe fallback
                # Fallback: Just all large numbers
                return sorted(large_nums)
                
    return result
# EVOLVE-BLOCK-END
