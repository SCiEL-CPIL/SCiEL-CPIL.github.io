import math
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # 1. Sieve primes up to 10^6
    limit = 1000000
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, end):
        """Compute phi(n) for all n in [start, end] using vectorized sieve."""
        size = end - start + 1
        vals = np.arange(start, end + 1, dtype=np.int64)
        phi = vals.copy()
        rem = vals.copy()
        
        max_p = int(math.isqrt(end))
        
        for p in primes:
            if p > max_p:
                break
            mask = rem % p == 0
            if not mask.any():
                continue
            
            phi[mask] -= phi[mask] // p
            idxs = np.where(mask)[0]
            rem[idxs] //= p
            
            while True:
                still_div = rem[idxs] % p == 0
                if not still_div.any():
                    break
                sub_idxs = idxs[still_div]
                rem[sub_idxs] //= p
                idxs = sub_idxs
        
        mask = rem > 1
        if mask.any():
            phi[mask] -= phi[mask] // rem[mask]
            
        return phi

    def find_longest_run(phi_vals):
        """Find longest run of consecutive integers with distinct totients."""
        best_k = 1
        best_s = 0
        n = len(phi_vals)
        
        seen = set()
        current_len = 0
        current_start = 0
        
        for i in range(n):
            val = phi_vals[i]
            if val in seen:
                seen = set()
                current_start = i
                current_len = 0
            seen.add(val)
            current_len += 1
            if current_len > best_k:
                best_k = current_len
                best_s = current_start
            if best_k >= 200:
                break
                
        return best_k, best_s

    best_k = 1
    best_s = 1

    # Strategy:
    # 1. Quick scan of small segments for baseline
    # 2. Random sampling across large range
    # 3. Refine around promising regions

    seg_size = 500000  # 500K per segment (reduced from 2M for faster per-segment cost)

    # Phase 1: Quick scan of a few segments
    for start in [1, 10**5, 10**6, 10**7]:
        if time.time() - start_time > deadline - 10:
            break
        seg_end = min(start + seg_size - 1, 10**12)
        if seg_end < start:
            continue
        try:
            phi_vals = compute_phi_segment(start, seg_end)
            k, s = find_longest_run(phi_vals)
            actual_s = start + s
            if k > best_k:
                best_k = k
                best_s = actual_s
                if best_k == 200:
                    return {'start': best_s, 'length': best_k}
        except Exception:
            pass

    # Phase 2: Random sampling at large values
    random.seed(42)
    low = 10**6
    high = 10**12 - seg_size

    for attempt in range(80):
        if time.time() - start_time > deadline - 5:
            break

        rand_start = random.randint(low, high)
        seg_end = min(rand_start + seg_size - 1, 10**12)

        try:
            phi_vals = compute_phi_segment(rand_start, seg_end)
            k, s = find_longest_run(phi_vals)
            actual_s = rand_start + s
            if k > best_k:
                best_k = k
                best_s = actual_s
                if best_k == 200:
                    return {'start': best_s, 'length': best_k}

                # Refine around promising region if found
                if k >= 3 and time.time() - start_time < deadline - 15:
                    refine_start = max(1, actual_s - 200000)
                    refine_end = min(10**12, actual_s + k + 200000)
                    refine_size = refine_end - refine_start + 1

                    if refine_size <= seg_size:
                        try:
                            phi_vals2 = compute_phi_segment(refine_start, refine_end)
                            k2, s2 = find_longest_run(phi_vals2)
                            actual_s2 = refine_start + s2
                            if k2 > best_k:
                                best_k = k2
                                best_s = actual_s2
                                if best_k == 200:
                                    return {'start': best_s, 'length': best_k}
                        except Exception:
                            pass
        except Exception:
            pass

    # Final safety checks
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1

    return {'start': best_s, 'length': best_k}
