import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Sieve primes up to 10^6
    limit = 1000000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.array([i for i, val in enumerate(is_prime) if val])

    best_k = 1
    best_s = 1

    # Sliding window state
    buffer_size = 200
    recent_phis = []  # list of phi values, index 0 is oldest
    left_idx = 0

    def update_window(idx, val):
        nonlocal best_k, best_s, recent_phis, left_idx

        # Find collision in recent_phis
        # recent_phis[i] corresponds to global index idx - len(recent_phis) + i
        collision_idx = -1
        for i in range(len(recent_phis) - 1, -1, -1):
            if recent_phis[i] == val:
                collision_idx = idx - len(recent_phis) + i
                break

        if collision_idx != -1:
            if collision_idx + 1 > left_idx:
                left_idx = collision_idx + 1

        current_len = idx - left_idx + 1
        if current_len > best_k:
            best_k = current_len
            best_s = left_idx

        recent_phis.append(val)
        if len(recent_phis) > buffer_size:
            recent_phis.pop(0)

    # Main loop
    segment_size = 500000
    start_val = 1
    while start_val <= 10**12:
        end_val = min(start_val + segment_size - 1, 10**12)
        n = end_val - start_val + 1

        # Segmented Sieve for phi
        vals = np.arange(start_val, end_val + 1, dtype=np.int64)
        phis = np.arange(start_val, end_val + 1, dtype=np.int64)

        for p in primes:
            if p * p > end_val:
                break
            first_mult = ((start_val + p - 1) // p) * p
            start_idx = first_mult - start_val
            step = p
            mask_indices = np.arange(start_idx, n, step)

            phi_slice = phis[mask_indices]
            phis[mask_indices] = phi_slice // p * (p - 1)

            v_slice = vals[mask_indices]
            for _ in range(40):
                mask_div = v_slice % p == 0
                if not np.any(mask_div):
                    break
                v_slice[mask_div] //= p
            vals[mask_indices] = v_slice

        mask_rem = vals > 1
        if np.any(mask_rem):
            phis[mask_rem] = phis[mask_rem] // vals[mask_rem] * (vals[mask_rem] - 1)

        # Vectorized sliding window update
        phis_list = phis.tolist()
        for i, val in enumerate(phis_list):
            idx = start_val + i
            update_window(idx, val)

        start_val = end_val + 1
        if time.time() > deadline:
            break

    # Guarantee valid result
    if best_s + best_k - 1 > 10**12:
        best_s = 1
        best_k = 1

    return {'start': best_s, 'length': best_k}
