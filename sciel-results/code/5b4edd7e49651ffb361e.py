"""Optimized program for large reciprocal weight under lcm separation."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 84.0  # 90s wall-clock, 6s safety margin

    N = 256
    nums = list(range(2, 257))
    n = len(nums)  # 255

    # Precompute conflict masks
    conflict_mask = [0] * n
    for i in range(n):
        a = nums[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            b = nums[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                mask |= (1 << j)
        conflict_mask[i] = mask

    # Weights
    weights = [Fraction(1, nums[i]) for i in range(n)]
    float_weights = [1.0 / nums[i] for i in range(n)]

    # Greedy Initialization
    def greedy_construct(ordering):
        selected_mask = 0
        for idx in ordering:
            if (selected_mask & conflict_mask[idx]) == 0:
                selected_mask |= (1 << idx)
        return selected_mask

    # Order by weight descending
    order = sorted(range(n), key=lambda i: -float_weights[i])
    
    best_mask = greedy_construct(order)
    best_score = sum(weights[i] for i in range(n) if (best_mask >> i) & 1)

    # Local Search
    def mask_to_set(m):
        s = set()
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            s.add(idx)
            m ^= lsb
        return s

    selected = mask_to_set(best_mask)
    current_score = best_score

    improved = True
    while improved and time.time() - start_time < DEADLINE - 15:
        improved = False
        
        # 1. Try adding
        for j in range(n):
            if j in selected:
                continue
            if (best_mask & conflict_mask[j]) == 0:
                best_mask |= (1 << j)
                selected.add(j)
                current_score += weights[j]
                improved = True
                break
        
        if time.time() - start_time > DEADLINE - 15:
            break

        # 2. Try 1-for-1 swap
        best_gain = -1.0
        best_swap = None
        for i in list(selected):
            for j in range(n):
                if j in selected:
                    continue
                # Check if j is compatible with selected \ {i}
                if (best_mask & conflict_mask[j] & ~(1 << i)) == 0:
                    gain = float_weights[j] - float_weights[i]
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (i, j)
        
        if best_swap and best_gain > 1e-15:
            i, j = best_swap
            best_mask &= ~(1 << i)
            best_mask |= (1 << j)
            selected.remove(i)
            selected.add(j)
            current_score += weights[j] - weights[i]
            improved = True

        # 3. Try 2-for-1 swap
        if time.time() - start_time < DEADLINE - 25:
            best_gain = -1.0
            best_swap = None
            sel_list = list(selected)
            for idx_i1 in range(len(sel_list)):
                if time.time() - start_time > DEADLINE - 25:
                    break
                i1 = sel_list[idx_i1]
                for idx_i2 in range(idx_i1 + 1, len(sel_list)):
                    i2 = sel_list[idx_i2]
                    mask_without = best_mask & ~(1 << i1) & ~(1 << i2)
                    for j in range(n):
                        if j in selected:
                            continue
                        if (mask_without & conflict_mask[j]) == 0:
                            gain = float_weights[j] - float_weights[i1] - float_weights[i2]
                            if gain > best_gain:
                                best_gain = gain
                                best_swap = (i1, i2, j)
            
            if best_swap and best_gain > 1e-15:
                i1, i2, j = best_swap
                best_mask &= ~(1 << i1)
                best_mask &= ~(1 << i2)
                best_mask |= (1 << j)
                selected.remove(i1)
                selected.remove(i2)
                selected.add(j)
                current_score += weights[j] - weights[i1] - weights[i2]
                improved = True

        # 4. Try 1-for-2 swap
        if time.time() - start_time < DEADLINE - 20:
            best_gain = -1.0
            best_swap = None
            for i in list(selected):
                mask_without = best_mask & ~(1 << i)
                # Find best pair j, k
                # To speed up, only consider j, k that are compatible with mask_without
                # and compatible with each other
                # This is expensive. Limit to top candidates by weight.
                candidates = []
                for j in range(n):
                    if j in selected:
                        continue
                    if (mask_without & conflict_mask[j]) == 0:
                        candidates.append(j)
                
                candidates.sort(key=lambda x: -float_weights[x])
                candidates = candidates[:20] # Limit candidates
                
                for idx_j1 in range(len(candidates)):
                    j1 = candidates[idx_j1]
                    for idx_j2 in range(idx_j1 + 1, len(candidates)):
                        j2 = candidates[idx_j2]
                        if conflict_mask[j1] & (1 << j2):
                            continue
                        gain = float_weights[j1] + float_weights[j2] - float_weights[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (i, j1, j2)
            
            if best_swap and best_gain > 1e-15:
                i, j1, j2 = best_swap
                best_mask &= ~(1 << i)
                best_mask |= (1 << j1)
                best_mask |= (1 << j2)
                selected.remove(i)
                selected.add(j1)
                selected.add(j2)
                current_score += weights[j1] + weights[j2] - weights[i]
                improved = True

    # Update global best
    if current_score > best_score:
        best_score = current_score
        best_mask = best_mask

    # B&B if time permits
    if time.time() - start_time < DEADLINE - 5:
        bnb_best_score = best_score
        bnb_best_mask = best_mask
        
        # Precompute suffix sums of float weights for the sorted order
        bnb_order = sorted(range(n), key=lambda i: -float_weights[i])
        suffix_sum = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + float_weights[bnb_order[i]]

        def dfs_bnb(idx, current_mask, current_score_float):
            nonlocal bnb_best_score, bnb_best_mask
            
            if time.time() - start_time > DEADLINE - 2:
                return
            
            # Bound: current + suffix_sum[idx]
            # This is loose (ignores conflicts), but fast.
            if current_score_float + suffix_sum[idx] <= float(bnb_best_score) + 1e-15:
                return

            if idx == n:
                if current_score_float > float(bnb_best_score) - 1e-15:
                    # Verify
                    exact_score = Fraction(0)
                    m = current_mask
                    while m:
                        lsb = m & (-m)
                        idx_m = lsb.bit_length() - 1
                        exact_score += weights[idx_m]
                        m ^= lsb
                    if exact_score > bnb_best_score:
                        bnb_best_score = exact_score
                        bnb_best_mask = current_mask
                return

            j = bnb_order[idx]
            
            # Exclude
            dfs_bnb(idx + 1, current_mask, current_score_float)
            
            # Include
            if (current_mask & conflict_mask[j]) == 0:
                dfs_bnb(idx + 1, current_mask | (1 << j), current_score_float + float_weights[j])

        dfs_bnb(0, 0, 0.0)
        best_score = bnb_best_score
        best_mask = bnb_best_mask

    # Convert to list
    result = []
    m = best_mask
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        result.append(nums[idx])
        m ^= lsb
        
    return result
# EVOLVE-BLOCK-END
