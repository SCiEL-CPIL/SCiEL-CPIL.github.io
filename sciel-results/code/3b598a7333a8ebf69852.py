import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)  # 47
    
    # Split into two halves for meet-in-the-middle
    mid = n // 2  # 23
    left = list(range(mid))
    right = list(range(mid, n))
    
    # LCM of 1..48
    # LCM = 2^4 * 3^3 * 5^2 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    # = 16 * 27 * 25 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    lcm = 1
    for d in denominators:
        g = 1
        a, b = lcm, d
        while b:
            a, b = b, a % b
        g = a
        lcm = lcm * d // g
    
    # Scale: weight[i] = lcm // denominators[i]
    weights = [lcm // d for d in denominators]
    target = lcm  # represents 1.0
    
    # Generate all subset sums for left half
    left_size = len(left)
    left_sums = [0]
    left_masks = [0]
    for i in left:
        w = weights[i]
        curr_size = len(left_sums)
        for j in range(curr_size):
            left_sums.append(left_sums[j] + w)
            left_masks.append(left_masks[j] | (1 << (i - mid + left_size)))
    
    # Actually, let me redo this more carefully
    # Left indices are 0..22
    left_sums = [0]
    left_masks = [0]
    for i in range(left_size):
        w = weights[i]
        curr_size = len(left_sums)
        for j in range(curr_size):
            left_sums.append(left_sums[j] + w)
            left_masks.append(left_masks[j] | (1 << i))
    
    # Sort left sums
    left_pairs = sorted(zip(left_sums, left_masks))
    left_sums_sorted = [p[0] for p in left_pairs]
    left_masks_sorted = [p[1] for p in left_pairs]
    
    best_sum = 0
    best_left_mask = 0
    best_right_mask = 0
    
    # Generate all subset sums for right half
    right_size = len(right)
    # For each right subset, we want to find the largest left sum such that
    # left_sum + right_sum < target (strictly below 1)
    
    # We'll iterate through all right subsets
    right_sums = [0]
    right_masks = [0]
    for i in range(right_size):
        w = weights[mid + i]
        curr_size = len(right_sums)
        for j in range(curr_size):
            right_sums.append(right_sums[j] + w)
            right_masks.append(right_masks[j] | (1 << i))
    
    # For each right subset sum, find max left sum < (target - right_sum)
    # target - right_sum must be > 0 for strict inequality: left_sum < target - right_sum
    # We need left_sum <= target - right_sum - 1 (since all are integers)
    
    for rs, rm in zip(right_sums, right_masks):
        if rs >= target:
            continue
        # Find largest left sum < (target - rs)
        # i.e., left_sum <= target - rs - 1
        limit = target - rs - 1
        if limit < 0:
            continue
        # Binary search for rightmost element <= limit
        idx = bisect.bisect_right(left_sums_sorted, limit) - 1
        if idx >= 0:
            ls = left_sums_sorted[idx]
            total = ls + rs
            if total > best_sum:
                best_sum = total
                best_left_mask = left_masks_sorted[idx]
                best_right_mask = rm
    
    # Construct indices from masks
    result = []
    for i in range(left_size):
        if best_left_mask & (1 << i):
            result.append(i)
    for i in range(right_size):
        if best_right_mask & (1 << i):
            result.append(mid + i)
    
    if not result:
        result = [0]
    
    return result
