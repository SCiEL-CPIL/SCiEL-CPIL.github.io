import time
import random
from math import gcd
import bisect
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = 30

    # Precompute LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)

    # W[i] = L // (i+1) for i in 0..29
    W = [L // (i + 1) for i in range(n)]
    W_np = np.array(W, dtype=np.int64)

    mid = n // 2  # 15

    # Generate Half A (indices 0 to 14) using vectorized NumPy
    def generate_half_np(start, end):
        length = end - start
        w = W_np[start:end]
        # Each index contributes -w, 0, +w
        # Build all 3^length combinations
        sums = np.zeros(1, dtype=np.int64)
        for i in range(length):
            wi = w[i]
            # For each existing sum, create three new sums
            new_sums = np.empty(len(sums) * 3, dtype=np.int64)
            new_sums[0::3] = sums - wi
            new_sums[1::3] = sums
            new_sums[2::3] = sums + wi
            sums = new_sums
        return sums

    # Generate all sums for half A
    sums_a = generate_half_np(0, mid)
    # Generate all sums for half B
    sums_b = generate_half_np(mid, n)

    # Sort Half B
    order_b = np.argsort(sums_b, kind='stable')
    arr_b_sums = sums_b[order_b]

    # For each element in B, we need to track which "pattern" it came from
    # Actually, for the MITM to work, we need to be able to reconstruct the full coefficient vector.
    # Let's use a different approach: store (sum, index_in_original) for B
    # Then decode the index to get the coefficients.
    
    # Actually, let's keep it simpler. We'll use the sorted sums and binary search.
    # For reconstruction, we need to know which specific B vector gives that sum.
    # Let's store the B vectors as well, or use a mapping.
    
    # Better approach: store B sums in a sorted array, and for each unique sum, 
    # store one representative index.
    
    # Let's use a dictionary: sum_value -> index in original B array
    # But there could be many duplicates. We just need one.
    b_sum_to_idx = {}
    for idx in range(len(sums_b)):
        s = sums_b[idx]
        if s not in b_sum_to_idx:
            b_sum_to_idx[s] = idx

    # Decode function
    def decode_from_index(idx, start, length):
        coeffs = [0] * length
        for j in range(length):
            digit = (idx // (3 ** j)) % 3
            coeffs[j] = digit - 1
        return coeffs

    best_abs_K = float('inf')
    best_coeffs = None

    # For each A sum, find closest B sum that makes total nonzero
    for i in range(len(sums_a)):
        if time.time() > deadline:
            break

        a_sum = sums_a[i]
        target = -a_sum

        # Binary search in sorted B sums
        idx = bisect.bisect_left(arr_b_sums, target)

        candidates = []

        # Check idx and idx-1
        if idx < len(arr_b_sums):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)

        # If exact match found, we need to skip it and check neighbors
        if idx < len(arr_b_sums) and arr_b_sums[idx] == target:
            # Find next > target
            j = idx
            while j < len(arr_b_sums) and arr_b_sums[j] == target:
                j += 1
            if j < len(arr_b_sums):
                candidates.append(j)

            # Find prev < target
            j = idx - 1
            while j >= 0 and arr_b_sums[j] == target:
                j -= 1
            if j >= 0:
                candidates.append(j)

        # Remove duplicates
        candidates = list(set(candidates))

        for c_idx in candidates:
            if 0 <= c_idx < len(arr_b_sums):
                b_sum_val = arr_b_sums[c_idx]
                total = a_sum + b_sum_val
                if total != 0:
                    abs_total = abs(total)
                    if abs_total < best_abs_K:
                        best_abs_K = abs_total
                        # Reconstruct coefficients
                        a_idx = i
                        b_idx = b_sum_to_idx.get(b_sum_val, 0)
                        # Find the actual B index that has this sum value
                        # b_sum_to_idx gives us one such index
                        coeffs_a = decode_from_index(a_idx, 0, mid)
                        coeffs_b = decode_from_index(b_idx, mid, mid)
                        best_coeffs = coeffs_a + coeffs_b

    if best_coeffs is None:
        # Fallback
        best_coeffs = [1] + [-1] + [0]*28
        best_abs_K = abs(W[0] - W[1])

    # Local Search with improved neighborhood
    random.seed(42)
    current = list(best_coeffs)
    current_K = sum(current[i] * W[i] for i in range(n))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(n))
        random.shuffle(indices)

        for i in indices:
            if time.time() > deadline:
                break

            old_c = current[i]

            # Try single flips
            for new_c in [-1, 0, 1]:
                if new_c == old_c:
                    continue

                new_K = current_K + (new_c - old_c) * W[i]

                if new_K == 0:
                    continue

                new_abs_K = abs(new_K)
                if new_abs_K < best_abs_K:
                    current[i] = new_c
                    current_K = new_K
                    best_abs_K = new_abs_K
                    best_coeffs = list(current)
                    improved = True
                    old_c = new_c
                    break  # Move to next index

            # Try two-swap moves (flip two positions simultaneously)
            if time.time() < deadline - 2:
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    old_cj = current[j]
                    for new_ci in [-1, 0, 1]:
                        if new_ci == old_c:
                            continue
                        for new_cj in [-1, 0, 1]:
                            if new_cj == old_cj:
                                continue
                            delta = (new_ci - old_c) * W[i] + (new_cj - old_cj) * W[j]
                            new_K = current_K + delta
                            if new_K == 0:
                                continue
                            new_abs_K = abs(new_K)
                            if new_abs_K < best_abs_K:
                                current[i] = new_ci
                                current[j] = new_cj
                                current_K = new_K
                                best_abs_K = new_abs_K
                                best_coeffs = list(current)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

    # Additional random restarts with local search
    if time.time() < deadline - 5:
        for restart in range(300):
            if time.time() > deadline:
                break

            current = list(best_coeffs)
            current_K = sum(current[i] * W[i] for i in range(n))

            # Perturb
            num_perturb = random.randint(1, 5)
            for _ in range(num_perturb):
                idx = random.randint(0, n - 1)
                current[idx] = random.choice([-1, 0, 1])

            current_K = sum(current[i] * W[i] for i in range(n))
            if current_K == 0:
                continue

            # Local search
            improved_local = True
            while improved_local and time.time() < deadline:
                improved_local = False
                indices = list(range(n))
                random.shuffle(indices)

                for i in indices:
                    if time.time() > deadline:
                        break

                    old_c = current[i]

                    for new_c in [-1, 0, 1]:
                        if new_c == old_c:
                            continue

                        new_K = current_K + (new_c - old_c) * W[i]

                        if new_K == 0:
                            continue

                        new_abs_K = abs(new_K)
                        if new_abs_K < best_abs_K:
                            current[i] = new_c
                            current_K = new_K
                            best_abs_K = new_abs_K
                            best_coeffs = list(current)
                            improved_local = True
                            old_c = new_c
                            break

    return best_coeffs
