import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 150s limit

    # Sieve primes up to 10^6
    limit = 1000000
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, end):
        """Compute phi for a segment [start, end] using segmented sieve approach."""
        size = end - start + 1
        if size <= 0:
            return np.array([], dtype=np.int64)
        
        # Initialize phi and remainder arrays
        vals = start + np.arange(size, dtype=np.int64)
        phi = vals.copy()
        rem = vals.copy()
        
        # Process primes in batches for better cache performance
        for p in primes:
            if p > end:
                break
            # Find indices where rem is divisible by p
            mask = (rem % p) == 0
            if not mask.any():
                continue
            # Update phi: phi[n] -= phi[n] / p
            phi[mask] -= phi[mask] // p
            # Remove all factors of p from rem
            rem[mask] //= p
            while True:
                still_div = (rem[mask] % p) == 0
                if not still_div.any():
                    break
                # Get indices that are still divisible
                idxs = np.where(mask)[0]
                sub_idx = idxs[still_div]
                rem[sub_idx] //= p
                mask = np.zeros(size, dtype=bool)
                mask[sub_idx] = True
        
        # Handle remaining prime factors > sqrt(end)
        mask = rem > 1
        if mask.any():
            phi[mask] -= phi[mask] // rem[mask]
            
        return phi

    def find_longest_run(phi_vals):
        """Find longest run of distinct phi values using sliding window."""
        n = len(phi_vals)
        if n <= 1:
            return 1, 0
        
        best_k = 1
        best_s = 0
        seen = {}
        left = 0
        
        for right in range(n):
            val = int(phi_vals[right])
            if val in seen and seen[val] >= left:
                left = seen[val] + 1
            seen[val] = right
            current_len = right - left + 1
            if current_len > best_k:
                best_k = current_len
                best_s = left
            if best_k >= 200:
                break
                
        return best_k, best_s

    best_k = 1
    best_s = 1

    # Phase 1: Quick check on small range to establish baseline
    if time.time() - start_time < deadline - 20:
        phi_vals = compute_phi_segment(1, 100000)
        k, s = find_longest_run(phi_vals)
        if k > best_k:
            best_k = k
            best_s = 1 + s

    # Phase 2: Search in promising regions near 10^12
    # Larger numbers tend to have more distinct phi values
    seg_size = 500000
    
    # Generate candidate starting points focused on high ranges
    candidates = []
    
    # Near 10^12 - most promising region
    for offset in [0, 1, 2, 5, 10, 20, 50, 100]:
        base = 10**12 - offset * seg_size
        if base > 1 and base + seg_size - 1 <= 10**12:
            candidates.append(base)
    
    # Powers of 10 and nearby
    for exp in range(8, 13):
        base = 10**exp
        if base + seg_size - 1 <= 10**12:
            candidates.append(base)
        if base - seg_size//2 > 1:
            candidates.append(base - seg_size//2)
    
    # Logarithmically spaced points
    for i in range(15):
        base = int(10**(7 + i*0.35))
        if base + seg_size - 1 <= 10**12:
            candidates.append(base)
    
    # Sort and deduplicate
    candidates = sorted(set(candidates))
    
    for base in candidates:
        if time.time() - start_time > deadline - 10:
            break
        seg_start = max(1, base)
        seg_end = min(seg_start + seg_size - 1, 10**12)
        if seg_start >= seg_end:
            continue
        try:
            phi_vals = compute_phi_segment(seg_start, seg_end)
            k, s = find_longest_run(phi_vals)
            actual_s = seg_start + s
            if k > best_k:
                best_k = k
                best_s = actual_s
                # If we found a good run, try to extend it
                if best_k >= 40 and time.time() - start_time < deadline - 15:
                    # Try extending in both directions
                    for direction in [-1, 1]:
                        new_start = best_s + direction * best_k
                        if new_start < 1 or new_start + seg_size > 10**12:
                            continue
                        new_seg_start = max(1, new_start)
                        new_seg_end = min(new_seg_start + seg_size - 1, 10**12)
                        if new_seg_start >= new_seg_end:
                            continue
                        try:
                            phi_ext = compute_phi_segment(new_seg_start, new_seg_end)
                            k_ext, s_ext = find_longest_run(phi_ext)
                            actual_s_ext = new_seg_start + s_ext
                            if k_ext > best_k:
                                best_k = k_ext
                                best_s = actual_s_ext
                        except Exception:
                            pass
        except Exception:
            pass

    # Phase 3: If time remains, try more segments
    if time.time() - start_time < deadline - 15 and best_k < 50:
        # Try segments at various locations
        small_seg = 200000
        for i in range(20):
            if time.time() - start_time > deadline - 8:
                break
            # Spread across the range
            base = int(10**(6 + i*0.4))
            if base > 10**12:
                break
            seg_start = base
            seg_end = min(base + small_seg - 1, 10**12)
            if seg_start < 1:
                continue
            try:
                phi_vals = compute_phi_segment(seg_start, seg_end)
                k, s = find_longest_run(phi_vals)
                actual_s = seg_start + s
                if k > best_k:
                    best_k = k
                    best_s = actual_s
            except Exception:
                pass

    # Final validation and clamping
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = max(1, 10**12 - best_s + 1)

    return {'start': best_s, 'length': best_k}
