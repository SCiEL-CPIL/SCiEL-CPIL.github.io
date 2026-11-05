import time
from math import gcd
import numpy as np

def construct():
    start_time = time.time()
    deadline = 75.0  # Safety margin before 90s kill

    n = 30

    # 1. Compute L = lcm(1, ..., 30)
    def lcm(a, b):
        return a * b // gcd(a, b)

    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)

    # 2. Compute weights w_i = L // i for i = 1..n
    w = [0] * (n + 1)
    for i in range(1, n + 1):
        w[i] = L // i

    # 3. Meet-in-the-middle: split indices 1..30 into two halves
    mid = n // 2  # 15
    left_indices = list(range(1, mid + 1))
    right_indices = list(range(mid + 1, n + 1))

    left_w = np.array([w[i] for i in left_indices], dtype=np.int64)
    right_w = np.array([w[i] for i in right_indices], dtype=np.int64)

    num_left = len(left_w)
    num_right = len(right_w)

    # Efficient sum generation using iterative expansion
    def generate_sums(weights):
        sums = np.array([0], dtype=np.int64)
        for w_val in weights:
            sums = np.concatenate([sums - w_val, sums, sums + w_val])
        return sums

    # Decode index from the iterative concat structure
    # c_i corresponds to digit i (0-indexed from MSB) in base 3 representation of index
    # Digit 0 -> -1, 1 -> 0, 2 -> 1
    def decode_concat_index(idx, n):
        coeffs = [0] * n
        for i in range(n):
            val = (idx // (3 ** (n - 1 - i))) % 3
            if val == 0:
                coeffs[i] = -1
            elif val == 1:
                coeffs[i] = 0
            else:
                coeffs[i] = 1
        return coeffs

    # Generate sums
    left_sums = generate_sums(left_w)
    
    if time.time() - start_time > deadline - 10:
        return [1] + [0]*29

    right_sums = generate_sums(right_w)
    
    if time.time() - start_time > deadline - 5:
        return [1] + [0]*29

    # Sort left sums
    sort_idx = np.argsort(left_sums)
    sorted_left_sums = left_sums[sort_idx]

    # Find best match
    targets = -right_sums
    insert_pts = np.searchsorted(sorted_left_sums, targets, side='left')

    best_abs = np.iinfo(np.int64).max
    best_left_idx = -1
    best_right_idx = -1

    # Check insert_pts
    valid_a = (insert_pts >= 0) & (insert_pts < len(sorted_left_sums))
    if np.any(valid_a):
        idx_a = insert_pts[valid_a]
        sums_a = sorted_left_sums[idx_a]
        totals_a = right_sums[valid_a] + sums_a
        nonzero_a = totals_a != 0
        if np.any(nonzero_a):
            abs_a = np.abs(totals_a[nonzero_a])
            min_pos = np.argmin(abs_a)
            min_val = abs_a[min_pos]
            if min_val < best_abs:
                valid_idx_in_batch = np.where(valid_a)[0]
                nonzero_positions = np.where(nonzero_a)[0]
                actual_right_idx = valid_idx_in_batch[nonzero_positions[min_pos]]
                actual_sorted_idx = idx_a[nonzero_a][min_pos]
                best_abs = min_val
                best_left_idx = sort_idx[actual_sorted_idx]
                best_right_idx = actual_right_idx

    # Check insert_pts - 1
    valid_b = (insert_pts > 0) & (insert_pts - 1 < len(sorted_left_sums))
    if np.any(valid_b):
        idx_b = insert_pts[valid_b] - 1
        sums_b = sorted_left_sums[idx_b]
        totals_b = right_sums[valid_b] + sums_b
        nonzero_b = totals_b != 0
        if np.any(nonzero_b):
            abs_b = np.abs(totals_b[nonzero_b])
            min_pos = np.argmin(abs_b)
            min_val = abs_b[min_pos]
            if min_val < best_abs:
                valid_idx_in_batch = np.where(valid_b)[0]
                nonzero_positions = np.where(nonzero_b)[0]
                actual_right_idx = valid_idx_in_batch[nonzero_positions[min_pos]]
                actual_sorted_idx = idx_b[nonzero_b][min_pos]
                best_abs = min_val
                best_left_idx = sort_idx[actual_sorted_idx]
                best_right_idx = actual_right_idx

    result = [0] * n

    if best_left_idx >= 0 and best_right_idx >= 0:
        left_coeffs = decode_concat_index(best_left_idx, num_left)
        right_coeffs = decode_concat_index(best_right_idx, num_right)
        
        for i in range(num_left):
            result[left_indices[i] - 1] = left_coeffs[i]
        for i in range(num_right):
            result[right_indices[i] - 1] = right_coeffs[i]
            
        # Local search refinement with exact integers
        current_int_sum = sum(result[i] * w[i+1] for i in range(n))
        
        improved = True
        while improved and time.time() - start_time < deadline - 2:
            improved = False
            for i in range(n):
                old_val = result[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    result[i] = new_val
                    new_int_sum = current_int_sum + (new_val - old_val) * w[i+1]
                    if new_int_sum != 0 and abs(new_int_sum) < abs(current_int_sum):
                        current_int_sum = new_int_sum
                        improved = True
                        break
                    else:
                        result[i] = old_val
                if improved:
                    break
                    
        return result

    # Fallback
    return [1] + [0]*29
