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

    def compute_phi_segment_fast(start, end):
        """Compute phi for a segment using vectorized operations."""
        size = end - start + 1
        if size <= 0:
            return np.array([], dtype=np.int64)
        
        # Initialize phi and remainder arrays
        idx = np.arange(size, dtype=np.int64)
        vals = start + idx
        phi = vals.copy()
        rem = vals.copy()
        
        # For each prime, process in batch
        for p in primes:
            if p > end:
                break
            # Find indices where rem is divisible by p
            # We need to be careful with large values
            if p * p > end:
                # Only one factor of p possible
                mask = rem % p == 0
                if mask.any():
                    phi[mask] -= phi[mask] // p
                    rem[mask] //= p
            else:
                mask = rem % p == 0
                if mask.any():
                    phi[mask] -= phi[mask] // p
                    idxs = np.where(mask)[0]
                    rem[idxs] //= p
                    # Remove remaining factors
                    while True:
                        still = rem[idxs] % p == 0
                        if not still.any():
                            break
                        idxs = idxs[still]
                        rem[idxs] //= p
        
        # Handle remaining prime factors > sqrt(end)
        mask = rem > 1
        if mask.any():
            phi[mask] -= phi[mask] // rem[mask]
            
        return phi

    def find_longest_run_fast(phi_vals):
        """Find longest run of distinct phi values."""
        best_k = 1
        best_s = 0
        n = len(phi_vals)
        
        if n <= 1:
            return best_k, best_s
        
        # Use a sliding window approach with a set
        seen = set()
        left = 0
        current_len = 0
        
        for right in range(n):
            val = phi_vals[right]
            while val in seen:
                seen.discard(phi_vals[left])
                left += 1
            seen.add(val)
            current_len = right - left + 1
            if current_len > best_k:
                best_k = current_len
                best_s = left
            if best_k >= 200:
                break
                
        return best_k, best_s

    best_k = 1
    best_s = 1

    # Phase 1: Quick check on small range
    if time.time() - start_time < deadline - 15:
        phi_vals = compute_phi_segment_fast(1, 100000)
        k, s = find_longest_run_fast(phi_vals)
        if k > best_k:
            best_k = k
            best_s = 1 + s

    # Phase 2: Search in promising regions
    # Larger numbers tend to have more distinct phi values
    seg_size = 200000
    
    # Generate candidate starting points
    candidates = []
    
    # Powers of 10 and nearby
    for exp in range(6, 13):
        base = 10**exp
        if base + seg_size - 1 <= 10**12:
            candidates.append(base)
        # Also try base - seg_size/2
        if base - seg_size//2 > 1:
            candidates.append(base - seg_size//2)
    
    # Near 10^12
    candidates.append(10**12 - seg_size)
    candidates.append(10**12 - 2*seg_size)
    
    # Some random-ish points in between
    for i in range(20):
        base = int(10**(6 + i*0.3))
        if base + seg_size - 1 <= 10**12:
            candidates.append(base)
    
    # Sort and deduplicate
    candidates = sorted(set(candidates))
    
    for base in candidates:
        if time.time() - start_time > deadline - 8:
            break
        seg_start = max(1, base)
        seg_end = min(seg_start + seg_size - 1, 10**12)
        if seg_start >= seg_end:
            continue
        try:
            phi_vals = compute_phi_segment_fast(seg_start, seg_end)
            k, s = find_longest_run_fast(phi_vals)
            actual_s = seg_start + s
            if k > best_k:
                best_k = k
                best_s = actual_s
                # If we found a good run, try to extend it
                if best_k >= 50 and time.time() - start_time < deadline - 10:
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
                            phi_ext = compute_phi_segment_fast(new_seg_start, new_seg_end)
                            k_ext, s_ext = find_longest_run_fast(phi_ext)
                            actual_s_ext = new_seg_start + s_ext
                            if k_ext > best_k:
                                best_k = k_ext
                                best_s = actual_s_ext
                        except Exception:
                            pass
        except Exception:
            pass

    # Phase 3: If time remains and best_k is still small, try more segments
    if time.time() - start_time < deadline - 12 and best_k < 30:
        # Try smaller segments at more locations
        small_seg = 50000
        for i in range(30):
            if time.time() - start_time > deadline - 5:
                break
            base = 10**6 * (i + 1)
            if base > 10**12:
                break
            seg_start = base
            seg_end = min(base + small_seg - 1, 10**12)
            if seg_start < 1:
                continue
            try:
                phi_vals = compute_phi_segment_fast(seg_start, seg_end)
                k, s = find_longest_run_fast(phi_vals)
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
