import numpy as np
import time

def construct():
    t_start = time.time()
    deadline = t_start + 85.0

    N = 10_000_000

    # Step 1: Sieve for smallest prime factor (vectorized)
    spf = np.zeros(N + 1, dtype=np.int64)
    spf[2:] = np.arange(2, N + 1)
    limit = int(np.sqrt(N))
    for i in range(2, limit + 1):
        if spf[i] == i:
            start = i * i
            spf[start::i] = np.where(spf[start::i] == 0, i, spf[start::i])
            # Actually we need: for j in range(i*i, N+1, i): if spf[j]==0: spf[j]=i
            # Vectorized:
            idx = np.arange(i * i, N + 1, i)
            mask = spf[idx] == 0
            if mask.any():
                spf[idx[mask]] = i

    if time.time() > deadline - 30.0:
        return [1, 1]

    # Step 2: Compute sigma(n) using multiplicative property
    sigma_arr = np.ones(N + 1, dtype=np.int64)
    p_pow = np.ones(N + 1, dtype=np.int64)
    p_sum = np.ones(N + 1, dtype=np.int64)

    for n in range(2, N + 1):
        if time.time() > deadline - 25.0:
            break
        p = spf[n]
        m = n // p
        if m % p == 0:
            p_pow[n] = p_pow[m] * p
            p_sum[n] = p_sum[m] + p_pow[n]
            sigma_arr[n] = sigma_arr[m] * p_sum[n] // p_sum[m]
        else:
            p_pow[n] = p
            p_sum[n] = 1 + p
            sigma_arr[n] = sigma_arr[m] * (1 + p)

    if time.time() > deadline - 10.0:
        return [1, 1]

    # Step 3: Sort by sigma value, then by n
    indices = np.arange(1, N + 1, dtype=np.int64)
    keys = sigma_arr[indices]
    order = np.lexsort((indices, keys))
    sorted_n = indices[order]
    sorted_s = keys[order]

    # Step 4: Build range mapping (sigma -> (start, end) in sorted arrays)
    # Find boundaries where sigma changes
    change = np.where(sorted_s[1:] != sorted_s[:-1])[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(sorted_s)]))
    
    # Build dict: sigma_value -> (start, end)
    sigma_to_range = {}
    for i in range(len(starts)):
        s_val = sorted_s[starts[i]]
        sigma_to_range[s_val] = (starts[i], ends[i])

    if time.time() > deadline - 5.0:
        return [1, 1]

    # Step 5a: Coarse vectorized search
    # For each b, find the position in sorted_n where 1.5*b would be
    b_arr = np.arange(1, N + 1, dtype=np.int64)
    targets = (3 * b_arr) // 2  # floor(1.5*b)
    targets_f = (3.0 * b_arr.astype(np.float64)) / 2.0
    
    # Search in sorted_n for each target
    pos_right = np.searchsorted(sorted_n, targets_f, side='right')
    pos_left = np.searchsorted(sorted_n, targets_f, side='left')
    
    # Candidate a's: sorted_n[pos_left-1], sorted_n[pos_left], sorted_n[pos_right-1], sorted_n[pos_right]
    # We need to check if sigma(a) == sigma(b)
    
    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    # Check a few candidate positions for each b
    # For efficiency, check pos_left-1, pos_left, pos_right-1, pos_right
    candidate_offsets = [-2, -1, 0, 1, 2]
    
    # Vectorized check: for each b and each offset, get candidate a
    # Then verify sigma(a) == sigma(b)
    
    b_sigma = sigma_arr[b_arr]
    
    for off in candidate_offsets:
        pos = pos_right + off
        # Clamp
        pos = np.clip(pos, 1, N)  # 1-indexed into sorted_n (0-indexed array, so valid 0..N-1)
        pos = np.clip(pos, 0, N - 1)
        
        a_cand = sorted_n[pos]
        
        # Check sigma match
        a_sigma = sigma_arr[a_cand]
        match = (a_sigma == b_sigma) & (a_cand != b_arr) & (2 * a_cand != 3 * b_arr)
        
        if match.any():
            # Compute error for matched ones
            a_m = a_cand[match]
            b_m = b_arr[match]
            # err = |a/b - 3/2| = |2a - 3b| / (2b)
            num = np.abs(2 * a_m.astype(np.int64) - 3 * b_m.astype(np.int64))
            den = 2 * b_m.astype(np.int64)
            err_arr = num.astype(np.float64) / den.astype(np.float64)
            
            min_idx = np.argmin(err_arr)
            min_err = err_arr[min_idx]
            if min_err < best_err:
                best_err = min_err
                best_a = int(a_m[min_idx])
                best_b = int(b_m[min_idx])
        
        if time.time() > deadline - 3.0:
            break

    # Step 5b: Fine-grained polish on top candidates
    # Re-examine b values near the best found, checking more neighbors
    if best_b > 1 and time.time() < deadline - 1.0:
        s_best = sigma_arr[best_b]
        start, end = sigma_to_range.get(s_best, (0, 0))
        if end - start >= 2:
            sub_n = sorted_n[start:end]
            # Check a window of b values around best_b
            b_lo = max(1, best_b - 200000)
            b_hi = min(N, best_b + 200000)
            
            for b in range(b_lo, b_hi + 1):
                if time.time() > deadline - 0.5:
                    break
                s = sigma_arr[b]
                if s not in sigma_to_range:
                    continue
                st, en = sigma_to_range[s]
                if en - st < 2:
                    continue
                target_f = 1.5 * b
                sub = sorted_n[st:en]
                pos = np.searchsorted(sub, target_f)
                # Check a window of candidates
                lo = max(0, pos - 5)
                hi = min(len(sub), pos + 5)
                for idx in range(lo, hi):
                    a_int = int(sub[idx])
                    if a_int == b:
                        continue
                    if 2 * a_int == 3 * b:
                        continue
                    num = abs(2 * a_int - 3 * b)
                    den = 2 * b
                    err = num / den
                    if err < best_err:
                        best_err = err
                        best_a = a_int
                        best_b = b

    return [int(best_a), int(best_b)]
