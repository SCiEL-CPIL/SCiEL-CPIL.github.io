"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return elapsed() < deadline
    
    # Precompute primes up to 10^6 using simple sieve
    LIMIT = 10**6
    is_prime = bytearray(b'\x01') * (LIMIT + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, LIMIT + 1) if is_prime[i]]
    
    # Function to compute phi for a range [lo, hi] using segmented approach with NumPy
    def compute_phi_segment(lo, hi):
        """Compute phi values for all integers in [lo, hi]."""
        n = hi - lo + 1
        phi_vals = np.arange(lo, hi + 1, dtype=np.int64)
        
        # Process primes up to sqrt(hi)
        sqrt_hi = int(math.isqrt(hi))
        
        for p in primes:
            if p > sqrt_hi:
                break
            
            # Find first multiple of p in [lo, hi]
            first = ((lo + p - 1) // p) * p
            
            # Create indices for multiples of p
            # indices in phi_vals are 0..n-1
            # actual values are lo..hi
            # multiples are first, first+p, ...
            # indices: (first-lo), (first+p-lo), ...
            
            # We can use numpy slicing if the step is constant, but start/end vary.
            # Using arange is safer.
            indices = np.arange(first - lo, n, p)
            
            if len(indices) > 0:
                vals = phi_vals[indices]
                phi_vals[indices] = vals - (vals // p)
        
        # Handle remaining prime factors
        # If phi_vals[i] > 1, the remaining factor is prime
        mask = phi_vals > 1
        phi_vals[mask] -= 1
        
        return phi_vals
    
    # Function to find longest run of distinct phi values in a numpy array
    def find_longest_distinct_run(phi_arr, offset=0):
        """Find longest consecutive run where all phi values are distinct.
        Returns (best_k, best_s) where best_s is the actual start (offset + index)."""
        n = len(phi_arr)
        best_k = 1
        best_s = offset + 1
        seen = {}  # phi_value -> index in phi_arr
        left = 0
        
        # Convert to list for faster indexing in Python loop?
        # Or iterate over numpy array directly.
        # Numpy iteration is slower than list iteration in Python.
        # But converting 10^7 ints to list takes time.
        # Let's try iterating over the numpy array directly first.
        # Actually, phi_arr is int64. Iterating yields numpy.int64 objects.
        
        for right in range(n):
            pv = int(phi_arr[right])
            if pv in seen and seen[pv] >= left:
                left = seen[pv] + 1
            seen[pv] = right
            current_len = right - left + 1
            if current_len > best_k:
                best_k = current_len
                best_s = offset + left + 1
            if not time_ok():
                break
        
        return best_k, best_s
    
    best_k = 1
    best_s = 1
    
    # Strategy: Scan large segments near 10^12
    # Segment size 10^7
    SEG_SIZE = 10**7
    MAX_N = 10**12
    
    # We will scan segments starting from high values downwards or upwards?
    # Let's start from 10^11 and go up.
    # Or just pick a few large segments.
    
    # To maximize chance, we can scan a contiguous block of size ~10^8 or 10^9
    # But 10^9 is too big for one phi computation (memory/time).
    # 10^7 is ~1s.
    # 120s / 1s = 120 segments.
    # 120 * 10^7 = 1.2 * 10^9 coverage.
    
    # Let's define a list of segments to try.
    # Start with a high range.
    base = 10**11
    segments = []
    for i in range(20): # Try 20 segments of 10^7
        lo = base + i * SEG_SIZE
        hi = lo + SEG_SIZE - 1
        if hi > MAX_N:
            hi = MAX_N
            lo = max(1, hi - SEG_SIZE + 1)
        segments.append((lo, hi))
    
    # Also try some lower ranges just in case?
    # Smaller ranges might have shorter distinct runs.
    # Let's focus on large ranges.
    
    for lo, hi in segments:
        if not time_ok():
            break
        
        # Ensure we don't overlap or go out of bounds
        if lo < 1:
            continue
        if hi > MAX_N:
            hi = MAX_N
        if lo > hi:
            continue
            
        try:
            phi_vals = compute_phi_segment(lo, hi)
            k, s = find_longest_distinct_run(phi_vals, lo - 1)
            
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            pass
    
    # Verify the result
    if best_s < 1:
        best_s = 1
    if best_k < 1:
        best_k = 1
    if best_k > 200:
        best_k = 200
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
        if best_k < 1:
            best_s = 1
            best_k = 1
    
    return {'start': best_s, 'length': best_k}

# EVOLVE-BLOCK-END
