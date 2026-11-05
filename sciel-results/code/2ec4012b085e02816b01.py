import numpy as np
from math import lcm, gcd
import time
import sys
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 256
    nums = list(range(2, N + 1))
    
    # Split into Small [2, 128] and Large [129, 256]
    small_nums = [x for x in nums if x <= 128]
    large_nums = [x for x in nums if x > 128]
    
    n_small = len(small_nums)
    n_large = len(large_nums)
    
    # Map values to indices in small_nums
    small_val_to_idx = {v: i for i, v in enumerate(small_nums)}
    
    # Weights
    w_small = [1.0 / x for x in small_nums]
    w_large = [1.0 / x for x in large_nums]
    
    # Base weight of all Large numbers
    base_w_large = sum(w_large)
    
    # Conflict graph for Small numbers
    # conflict_s[i] is bitmask of j such that lcm(small_nums[i], small_nums[j]) <= 256
    conflict_s = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        bits = 0
        for j in range(n_small):
            if i == j:
                continue
            b = small_nums[j]
            g = gcd(a, b)
            if (a // g) * b <= 256:
                bits |= (1 << j)
        conflict_s[i] = bits
        
    # For each small number, find which large numbers it conflicts with
    # conflict_l[i] is bitmask of k in large_nums such that lcm(small_nums[i], large_nums[k]) <= 256
    conflict_l = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        bits = 0
        for k in range(n_large):
            b = large_nums[k]
            g = gcd(a, b)
            if (a // g) * b <= 256:
                bits |= (1 << k)
        conflict_l[i] = bits
        
    # Precompute weight of large numbers removed for a given set of small indices?
    # It's better to compute on the fly or use a helper.
    
    def penalty_for_small_set(mask_s):
        """Calculate sum of weights of large numbers conflicting with any in mask_s."""
        if mask_s == 0:
            return 0.0
        # Union of conflict_l[i] for all i in mask_s
        union_mask = 0
        m = mask_s
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            union_mask |= conflict_l[i]
            m ^= lsb
        
        # Sum weights of large numbers in union_mask
        s = 0.0
        u = union_mask
        while u:
            lsb = u & (-u)
            k = lsb.bit_length() - 1
            s += w_large[k]
            u ^= lsb
        return s

    def weight_for_small_set(mask_s):
        """Calculate sum of weights of small numbers in mask_s."""
        s = 0.0
        m = mask_s
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            s += w_small[i]
            m ^= lsb
        return s

    # Objective: Maximize weight_for_small_set(S) - penalty_for_small_set(S)
    # Subject to S being independent in conflict_s
    
    best_gain = -1e9
    best_mask_s = 0
    
    # Branch and Bound
    # State: current_mask, current_weight, current_penalty, available_mask
    # available_mask: bits that can still be added (not in current, not conflicting with current)
    
    import sys
    sys.setrecursionlimit(10000)
    
    def solve_bnb(available, current_mask, current_w, current_pen):
        nonlocal best_gain, best_mask_s
        
        if time.time() > deadline - 2.0:
            return
            
        # Calculate upper bound
        # Max possible additional weight from available
        # We can just sum weights of all in available (optimistic)
        # But we must subtract potential penalty.
        # Penalty is non-decreasing. So min penalty is current_pen.
        # Max weight is current_w + sum(weights in available).
        
        # Simple bound: current_w + sum(weights in available) - current_pen
        # To speed up, precompute prefix sums? Or just iterate.
        # n_small is 127. Iteration is fast.
        
        avail_sum = 0.0
        m = available
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            avail_sum += w_small[i]
            m ^= lsb
            
        upper_bound = current_w + avail_sum - current_pen
        
        if upper_bound <= best_gain + 1e-9:
            return
            
        if available == 0:
            gain = current_w - current_pen
            if gain > best_gain:
                best_gain = gain
                best_mask_s = current_mask
            return
            
        # Heuristic: Pick a node from available to branch on
        # Pick the one with highest weight? Or highest degree?
        # Let's pick the first one available for simplicity, or highest weight.
        # Highest weight is better for pruning.
        
        # Find index with max weight in available
        max_w = -1
        max_i = -1
        m = available
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            if w_small[i] > max_w:
                max_w = w_small[i]
                max_i = i
            m ^= lsb
            
        if max_i == -1:
            return
            
        # Branch 1: Include max_i
        new_mask = current_mask | (1 << max_i)
        new_w = current_w + w_small[max_i]
        
        # Update available: remove max_i and its neighbors
        new_available = available & ~(1 << max_i) & ~conflict_s[max_i]
        
        # Update penalty
        # Penalty is determined by union of conflict_l.
        # We need to calculate new penalty.
        # penalty(new_mask)
        new_pen = penalty_for_small_set(new_mask)
        
        solve_bnb(new_available, new_mask, new_w, new_pen)
        
        # Branch 2: Exclude max_i
        # Remove max_i from available
        new_available = available & ~(1 << max_i)
        solve_bnb(new_available, current_mask, current_w, current_pen)
        
    # Initial call
    # All small numbers are available
    initial_available = (1 << n_small) - 1
    solve_bnb(initial_available, 0, 0.0, 0.0)
    
    # Construct result
    # Start with all large numbers
    result_set = set(large_nums)
    
    # Add selected small numbers
    m = best_mask_s
    while m:
        lsb = m & (-m)
        i = lsb.bit_length() - 1
        result_set.add(small_nums[i])
        m ^= lsb
        
    # Remove large numbers that conflict with selected small numbers
    # We need to remove y in L if lcm(x,y) <= 256 for any x in S_S
    union_conflict_l = 0
    m = best_mask_s
    while m:
        lsb = m & (-m)
        i = lsb.bit_length() - 1
        union_conflict_l |= conflict_l[i]
        m ^= lsb
        
    u = union_conflict_l
    while u:
        lsb = u & (-u)
        k = lsb.bit_length() - 1
        result_set.remove(large_nums[k])
        u ^= lsb
        
    result = list(result_set)
    
    # Validation
    if len(result) > 0:
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if lcm(result[i], result[j]) <= 256:
                    # Fallback to safe set
                    return list(range(129, 257))
                    
    return result
