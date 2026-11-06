"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 256
    nums = list(range(2, N + 1))
    n = len(nums)
    
    # Precompute conflict matrix as bitmasks
    # conflict[i] has bit j set if lcm(nums[i], nums[j]) <= 256
    conflict = [0] * n
    for i in range(n):
        ai = nums[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            if lcm(ai, nums[j]) <= N:
                mask |= (1 << j)
        conflict[i] = mask
    
    weights = [1.0 / x for x in nums]
    
    def compute_score_bitmask(bm):
        total = 0.0
        temp = bm
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            total += weights[i]
            temp ^= lsb
        return total
    
    def is_valid_bitmask(bm):
        temp = bm
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            if conflict[i] & bm:
                return False
            temp ^= lsb
        return True
    
    # Split into small (2..16) and large (17..256)
    small_vals = list(range(2, 17))
    small_indices = [nums.index(x) for x in small_vals]
    n_small = len(small_indices)
    
    large_indices_list = [i for i in range(n) if nums[i] >= 17]
    
    best_bitmask = 0
    best_score = 0.0
    
    # Iterate over all subsets of small numbers (2^15 = 32768)
    for mask_small in range(1 << n_small):
        if time.time() > deadline - 5.0:
            break
        
        # Build small set bitmask
        small_bm = 0
        for i in range(n_small):
            if mask_small & (1 << i):
                small_bm |= (1 << small_indices[i])
        
        # Check internal validity of small set
        if not is_valid_bitmask(small_bm):
            continue
        
        # Determine compatible large numbers
        large_bm = 0
        for i in large_indices_list:
            large_bm |= (1 << i)
        
        # Remove conflicts with small set
        for i in range(n_small):
            if mask_small & (1 << i):
                si = small_indices[i]
                large_bm &= ~conflict[si]
        
        # Greedy selection of large numbers by weight
        active = large_bm
        selected_bm = 0
        while active:
            # Find highest weight active node
            best_i = -1
            best_w = -1.0
            temp = active
            while temp:
                lsb = temp & (-temp)
                i = lsb.bit_length() - 1
                if weights[i] > best_w:
                    best_w = weights[i]
                    best_i = i
                temp ^= lsb
            
            if best_i < 0:
                break
            
            selected_bm |= (1 << best_i)
            active &= ~conflict[best_i]
            active &= ~(1 << best_i)
        
        full_bm = small_bm | selected_bm
        score = compute_score_bitmask(full_bm)
        
        if score > best_score:
            best_score = score
            best_bitmask = full_bm
    
    # Local search: 1-for-1 swaps
    if time.time() < deadline - 10.0 and best_bitmask:
        working_bm = best_bitmask
        working_score = best_score
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            temp = working_bm
            while temp:
                lsb = temp & (-temp)
                rem_i = lsb.bit_length() - 1
                temp ^= lsb
                
                remaining_bm = working_bm & ~(1 << rem_i)
                
                # Try adding each candidate not in working set
                for add_i in range(n):
                    if working_bm & (1 << add_i):
                        continue
                    # Check if add_i is compatible with remaining
                    if conflict[add_i] & remaining_bm:
                        continue
                    new_bm = remaining_bm | (1 << add_i)
                    new_score = working_score - weights[rem_i] + weights[add_i]
                    if new_score > working_score + 1e-12:
                        working_bm = new_bm
                        working_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        if working_score > best_score + 1e-12:
            best_score = working_score
            best_bitmask = working_bm
    
    # Final polish: 2-for-1 and 1-for-2 swaps
    if time.time() < deadline - 3.0 and best_bitmask:
        working_bm = best_bitmask
        working_score = best_score
        improved = True
        
        while improved and time.time() < deadline - 3.0:
            improved = False
            
            # Try 2-for-1: remove two, add one
            temp = working_bm
            while temp and not improved:
                lsb1 = temp & (-temp)
                i1 = lsb1.bit_length() - 1
                temp2 = temp ^ lsb1
                while temp2 and not improved:
                    lsb2 = temp2 & (-temp2)
                    i2 = lsb2.bit_length() - 1
                    remaining_bm = working_bm & ~(1 << i1) & ~(1 << i2)
                    
                    for add_i in range(n):
                        if working_bm & (1 << add_i):
                            continue
                        if conflict[add_i] & remaining_bm:
                            continue
                        new_score = working_score - weights[i1] - weights[i2] + weights[add_i]
                        if new_score > working_score + 1e-12:
                            working_bm = remaining_bm | (1 << add_i)
                            working_score = new_score
                            improved = True
                            break
                    temp2 ^= lsb2
                temp ^= lsb1
            
            # Try 1-for-2: remove one, add two
            if not improved:
                temp = working_bm
                while temp and not improved:
                    lsb = temp & (-temp)
                    rem_i = lsb.bit_length() - 1
                    remaining_bm = working_bm & ~(1 << rem_i)
                    
                    best_pair = None
                    best_gain = 0.0
                    
                    for c1 in range(n):
                        if working_bm & (1 << c1):
                            continue
                        if conflict[c1] & remaining_bm:
                            continue
                        temp_bm = remaining_bm | (1 << c1)
                        for c2 in range(c1 + 1, n):
                            if working_bm & (1 << c2):
                                continue
                            if conflict[c2] & temp_bm:
                                continue
                            gain = weights[c1] + weights[c2] - weights[rem_i]
                            if gain > best_gain:
                                best_gain = gain
                                best_pair = (c1, c2)
                    
                    if best_pair is not None and best_gain > 1e-12:
                        working_bm = remaining_bm | (1 << best_pair[0]) | (1 << best_pair[1])
                        working_score += best_gain
                        improved = True
                    
                    temp ^= lsb
            
            if working_score > best_score + 1e-12:
                best_score = working_score
                best_bitmask = working_bm
    
    # Convert bitmask to result list
    result = []
    temp = best_bitmask
    while temp:
        lsb = temp & (-temp)
        i = lsb.bit_length() - 1
        result.append(nums[i])
        temp ^= lsb
    
    return result
# EVOLVE-BLOCK-END
