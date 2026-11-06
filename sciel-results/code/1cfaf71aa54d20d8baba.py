import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s limit

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # We need to find the longest run of consecutive integers with distinct totients.
    # Strategy:
    # 1. Use a segmented sieve to compute phi(n) for a large range efficiently.
    # 2. Use a sliding window to find the longest subsequence of distinct values.
    # 3. Since the range is up to 10^12, we can't compute phi for all of them.
    #    However, long runs of distinct totients are rare.
    #    The seed program found a run of length 27 in [1, 20000].
    #    We can search in larger ranges by computing phi for segments.
    #
    # Key insight: phi(n) is even for n > 2. So for any run of length > 2,
    # all phi values must be even and distinct.
    #
    # Let's try to search in a larger range, say up to 10^7 or 10^8, using a segmented sieve.
    # But 10^8 might be too slow in Python. Let's try 10^7 first.
    #
    # Actually, the observed score of 2.777... suggests that the best known run is about 27.78 * seed_length.
    # If the seed found length 27, then 2.777 * 27 ≈ 75. So there might be a run of length ~75.
    #
    # Let's use a segmented sieve to compute phi for a large segment, then use a sliding window.

    def compute_phi_segment(lo, hi):
        """Compute phi(n) for n in [lo, hi] using a segmented sieve."""
        n = hi - lo + 1
        phi = np.arange(lo, hi + 1, dtype=np.int64)
        
        # Sieve of Eratosthenes for primes up to sqrt(hi)
        limit = int(np.sqrt(hi)) + 1
        is_prime = np.ones(limit + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, limit + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        
        primes = np.where(is_prime)[0]
        
        for p in primes:
            if p * p > hi:
                break
            # Find the first multiple of p in [lo, hi]
            start = ((lo + p - 1) // p) * p
            if start < p * p:
                start = p * p
            
            # For each multiple of p in [lo, hi], update phi
            for m in range(start, hi + 1, p):
                idx = m - lo
                if phi[idx] % p == 0:
                    phi[idx] = phi[idx] // p * (p - 1)
        
        return phi

    def find_longest_distinct_run(phi_values, max_length=200):
        """Find the longest run of consecutive distinct values in phi_values."""
        n = len(phi_values)
        best_len = 0
        best_start = 0
        
        # Sliding window approach
        # We want the longest window [i, j] such that all phi_values[i..j] are distinct.
        # But we also want consecutive integers, so the window must be contiguous.
        
        # Use a dictionary to track the last seen index of each phi value.
        last_seen = {}
        left = 0
        
        for right in range(n):
            val = phi_values[right]
            if val in last_seen and last_seen[val] >= left:
                left = last_seen[val] + 1
            last_seen[val] = right
            
            current_len = right - left + 1
            if current_len > best_len:
                best_len = current_len
                best_start = left
        
        return best_start, best_len

    # Search in increasing segments
    # Start with a reasonable segment size
    segment_size = 10**7
    best_length = 1
    best_start = 1
    
    # First, try a small range to get a baseline
    if time_remaining() > 5:
        phi_small = compute_phi_segment(1, 200000)
        s, k = find_longest_distinct_run(phi_small)
        if k > best_length:
            best_length = k
            best_start = s + 1  # phi_small[0] corresponds to n=1
    
    # Now try larger segments
    # We'll try segments centered around different locations
    # Long runs might occur at specific locations. Let's try a few.
    
    # Try segments around powers of 2 or other interesting locations
    # Actually, let's just try a few large segments at different locations.
    
    locations = [
        10**6,
        5*10**6,
        10**7,
        5*10**7,
        10**8,
    ]
    
    for loc in locations:
        if time_remaining() < 10:
            break
        
        lo = max(1, loc - segment_size // 2)
        hi = min(10**12, loc + segment_size // 2)
        
        if hi - lo + 1 > segment_size:
            hi = lo + segment_size - 1
        
        if lo >= hi:
            continue
        
        try:
            phi_seg = compute_phi_segment(lo, hi)
            s, k = find_longest_distinct_run(phi_seg)
            if k > best_length:
                best_length = k
                best_start = lo + s
        except Exception:
            continue
    
    # If we still have time, try to refine around the best location found
    if time_remaining() > 30 and best_length > 1:
        # Search in a smaller window around best_start
        refine_lo = max(1, best_start - 1000)
        refine_hi = min(10**12, best_start + 1000 + best_length)
        
        try:
            phi_refine = compute_phi_segment(refine_lo, refine_hi)
            s, k = find_longest_distinct_run(phi_refine)
            if k > best_length:
                best_length = k
                best_start = refine_lo + s
        except Exception:
            pass
    
    # Ensure the result is valid
    if best_start + best_length - 1 > 10**12:
        best_length = min(best_length, 10**12 - best_start + 1)
    if best_length < 1:
        best_length = 1
    if best_start < 1:
        best_start = 1
    
    return {'start': best_start, 'length': best_length}
