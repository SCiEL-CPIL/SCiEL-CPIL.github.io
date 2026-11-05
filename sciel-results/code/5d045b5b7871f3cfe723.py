import time
from math import gcd
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

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

    # Generate all 3^15 left-half coefficient vectors and their sums
    num_left = len(left_w)
    total_left = 3 ** num_left

    # Use base-3 representation to enumerate all combinations
    # For each mask in [0, 3^15), compute the signed sum
    # Digit d in base-3: 0 -> -1, 1 -> 0, 2 -> +1

    # Vectorized generation using NumPy
    # Create all masks
    masks = np.arange(total_left, dtype=np.int64)

    # Compute digits for each position
    # For position j, digit = (mask // 3^j) % 3
    powers_of_3 = np.array([3 ** j for j in range(num_left)], dtype=np.int64)

    # For each position, compute the digit for all masks
    # digits[j] = (masks // powers_of_3[j]) % 3
    # Then convert: 0 -> -1, 1 -> 0, 2 -> +1
    # sum = sum_j digits[j] * left_w[j]

    # Build the sum vector
    left_sums = np.zeros(total_left, dtype=np.int64)
    # Also store the coefficient vectors for reconstruction
    # But storing 14M * 15 coefficients is too much memory
    # Instead, we'll store just the mask that achieves each sum
    # and reconstruct coefficients from the mask

    # For each position j, add contribution
    for j in range(num_left):
        digits = (masks // powers_of_3[j]) % 3
        # Convert: 0->-1, 1->0, 2->+1
        vals = np.where(digits == 0, -1, np.where(digits == 1, 0, 1))
        left_sums += vals * left_w[j]

    # Now left_sums[i] is the signed sum for the i-th combination
    # We want to find, for each right sum R, the left sum L that minimizes |L + R|, L+R != 0

    # Create a sorted array of (sum, mask) pairs
    # But we only need unique sums with their best (min abs) mask? No, we need all to find exact matches
    # Actually, for a given sum value, any mask achieving it works
    # So let's create a dict: sum_value -> mask (first one found)

    # But 14M entries in a dict is feasible
    # Let's create sorted arrays for binary search

    # Sort left_sums and keep track of original indices
    sort_idx = np.argsort(left_sums)
    sorted_sums = left_sums[sort_idx]
    sorted_masks = masks[sort_idx]

    # Now generate all right-half combinations and for each, binary search in sorted_sums
    num_right = len(right_w)
    total_right = 3 ** num_right

    best_abs = np.iinfo(np.int64).max
    best_left_mask = -1
    best_right_mask = -1

    # Process right combinations in batches to manage memory
    batch_size = 100000
    for batch_start in range(0, total_right, batch_size):
        batch_end = min(batch_start + batch_size, total_right)
        batch_count = batch_end - batch_start

        # Check time
        if time.time() - start_time > deadline:
            break

        batch_masks = np.arange(batch_start, batch_end, dtype=np.int64)

        # Compute right sums for this batch
        right_sums = np.zeros(batch_count, dtype=np.int64)
        for j in range(num_right):
            digits = (batch_masks // powers_of_3[j]) % 3
            vals = np.where(digits == 0, -1, np.where(digits == 1, 0, 1))
            right_sums += vals * right_w[j]

        # For each right sum, find best left sum
        # Target = -right_sum
        # Binary search in sorted_sums

        targets = -right_sums

        # Use np.searchsorted to find insertion points
        insert_pts = np.searchsorted(sorted_sums, targets, side='left')

        # Check candidates: insert_pts and insert_pts - 1
        # For each right sum, check up to 2 candidates

        # Candidate 1: sorted_sums[insert_pts]
        # Candidate 2: sorted_sums[insert_pts - 1]

        # Handle boundaries
        # For candidate 1 (right of target)
        valid1 = (insert_pts >= 0) & (insert_pts < len(sorted_sums))
        if np.any(valid1):
            cand1_sums = sorted_sums[insert_pts[valid1]]
            cand1_totals = right_sums[valid1] + cand1_sums
            # Exclude zero
            nonzero1 = cand1_totals != 0
            if np.any(nonzero1):
                abs1 = np.abs(cand1_totals[nonzero1])
                # Find min
                min_idx1 = np.argmin(abs1)
                min_val1 = abs1[min_idx1]
                if min_val1 < best_abs:
                    # Get the actual indices
                    valid_indices = np.where(valid1)[0]
                    actual_idx = valid_indices[min_idx1]
                    # But we need to map back: nonzero1 is a subset of valid1
                    # Let's redo this more carefully

        # Let me redo the candidate checking more carefully
        # For each right sum in the batch, check candidates

        # Create arrays for candidate evaluation
        # Candidate A: insert_pts (first >= target)
        # Candidate B: insert_pts - 1 (last < target)

        # Candidate A
        valid_a = (insert_pts >= 0) & (insert_pts < len(sorted_sums))
        if np.any(valid_a):
            idx_a = insert_pts[valid_a]
            sums_a = sorted_sums[idx_a]
            totals_a = right_sums[valid_a] + sums_a
            nonzero_a = totals_a != 0
            if np.any(nonzero_a):
                abs_a = np.abs(totals_a[nonzero_a])
                min_pos = np.argmin(abs_a)
                min_val = abs_a[min_pos]
                if min_val < best_abs:
                    # Find which right sum this corresponds to
                    valid_idx_in_batch = np.where(valid_a)[0]
                    nonzero_positions = np.where(nonzero_a)[0]
                    actual_batch_idx = valid_idx_in_batch[nonzero_positions[min_pos]]
                    best_abs = min_val
                    best_left_mask = sorted_masks[idx_a[nonzero_a[min_pos]]] if False else sorted_masks[idx_a[nonzero_positions[min_pos]]]
                    # Wait, need to be careful with indexing
                    # idx_a is indexed by valid_a positions
                    # nonzero_a is a boolean mask on idx_a
                    # So the actual index into sorted_sums is idx_a[nonzero_a][min_pos]
                    actual_sorted_idx = idx_a[nonzero_a][min_pos]
                    best_left_mask = sorted_masks[actual_sorted_idx]
                    best_right_mask = batch_masks[actual_batch_idx]

        # Candidate B
        valid_b = (insert_pts > 0) & (insert_pts - 1 < len(sorted_sums))
        if np.any(valid_b):
            idx_b = insert_pts[valid_b] - 1
            sums_b = sorted_sums[idx_b]
            totals_b = right_sums[valid_b] + sums_b
            nonzero_b = totals_b != 0
            if np.any(nonzero_b):
                abs_b = np.abs(totals_b[nonzero_b])
                min_pos = np.argmin(abs_b)
                min_val = abs_b[min_pos]
                if min_val < best_abs:
                    valid_idx_in_batch = np.where(valid_b)[0]
                    nonzero_positions = np.where(nonzero_b)[0]
                    actual_batch_idx = valid_idx_in_batch[nonzero_positions[min_pos]]
                    actual_sorted_idx = idx_b[nonzero_b][min_pos]
                    best_abs = min_val
                    best_left_mask = sorted_masks[actual_sorted_idx]
                    best_right_mask = batch_masks[actual_batch_idx]

    # If we found a solution, reconstruct coefficients
    if best_left_mask >= 0 and best_right_mask >= 0:
        result = [0] * n

        # Reconstruct left coefficients from best_left_mask
        temp = best_left_mask
        for j in range(num_left):
            digit = temp % 3
            temp //= 3
            if digit == 0:
                val = -1
            elif digit == 1:
                val = 0
            else:
                val = 1
            result[left_indices[j] - 1] = val

        # Reconstruct right coefficients from best_right_mask
        temp = best_right_mask
        for j in range(num_right):
            digit = temp % 3
            temp //= 3
            if digit == 0:
                val = -1
            elif digit == 1:
                val = 0
            else:
                val = 1
            result[right_indices[j] - 1] = val

        return result

    # Fallback: return a simple nonzero solution
    fallback = [0] * n
    fallback[0] = 1
    return fallback
