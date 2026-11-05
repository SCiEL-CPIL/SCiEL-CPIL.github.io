import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    # Generate primes up to 10^6 using sieve
    limit = 10**6
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # For each prime p, we want to find ranges [s, s+k-1] where phi values are distinct
    # Strategy: scan consecutive integers in windows, compute phi for each, track distinct runs
    
    # We'll use a segmented sieve approach for phi computation
    # For a window [lo, hi], compute phi for all integers in that range
    
    def compute_phi_segment(lo, hi):
        """Compute phi(n) for all n in [lo, hi] using segmented sieve."""
        size = hi - lo + 1
        phi = np.arange(lo, hi + 1, dtype=np.int64)
        # For each prime p, for multiples of p in [lo, hi], apply phi[m] -= phi[m] // p
        for p in primes:
            if p > hi:
                break
            # Find first multiple of p >= lo
            start = ((lo + p - 1) // p) * p
            if start < lo:
                start = lo
            # Apply phi update for all multiples of p
            for m in range(start, hi + 1, p):
                idx = m - lo
                phi[idx] -= phi[idx] // p
        return phi
    
    best_k = 1
    best_s = 1
    
    # Scan in windows of size 10000
    window_size = 10000
    scan_start = 1
    max_scan = 10**9  # scan up to 1 billion, which is within 10^12
    
    while scan_start <= max_scan and time_left() > 10.0:
        scan_end = min(scan_start + window_size - 1, max_scan)
        if scan_start >= 10**12 - 200:
            break
        
        phi_vals = compute_phi_segment(scan_start, scan_end)
        
        # Find longest run of distinct phi values
        seen = set()
        current_k = 0
        current_start = scan_start
        
        for i in range(len(phi_vals)):
            val = int(phi_vals[i])
            if val not in seen:
                seen.add(val)
                current_k += 1
            else:
                if current_k > best_k:
                    best_k = current_k
                    best_s = current_start
                # Reset
                seen = set()
                current_k = 1
                current_start = scan_start + i
                seen.add(val)
        
        if current_k > best_k:
            best_k = current_k
            best_s = current_start
        
        # Move to next window with overlap to catch runs spanning window boundaries
        scan_start = scan_end - 100  # overlap of 100 to catch boundary runs
        if scan_start < 1:
            scan_start = 1
    
    # Final check: ensure s + k - 1 <= 10^12
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    
    if best_k < 1:
        best_k = 1
        best_s = 1
    
    return {'start': best_s, 'length': best_k}
