import time
import math

def construct():
    start_time = time.time()
    deadline = 110  # 10 seconds safety margin
    
    MAX_N = 10**12
    MAX_K = 200
    
    # Step 1: Sieve primes up to sqrt(MAX_N) = 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    def compute_phi_segment(L, size):
        """
        Computes phi for range [L, L+size-1]
        Returns a list of phi values.
        """
        vals = [L + i for i in range(size)]
        phis = [L + i for i in range(size)]
        
        # Process primes
        # We only need primes p such that p*p <= max possible value in segment?
        # Actually, if p > sqrt(R), p can appear at most once.
        # So we process all primes <= sqrt(L+size-1)
        max_val = L + size - 1
        sqrt_max = int(math.isqrt(max_val))
        
        for p in primes:
            if p > sqrt_max:
                break
            # Find first multiple of p in [L, L+size-1]
            start = (L + p - 1) // p * p
            for j in range(start, L + size, p):
                idx = j - L
                if vals[idx] % p == 0:
                    phis[idx] = (phis[idx] // p) * (p - 1)
                    while vals[idx] % p == 0:
                        vals[idx] //= p
        
        # Handle remaining prime factors
        for i in range(size):
            if vals[i] > 1:
                phis[i] = (phis[i] // vals[i]) * (vals[i] - 1)
                
        return phis

    def find_max_distinct_run(phis):
        """
        Finds the longest run of distinct values in the list phis.
        Returns (length, start_index_in_phis)
        """
        last_seen = {}
        start_idx = 0
        max_len = 0
        max_start = 0
        
        for i in range(len(phis)):
            val = phis[i]
            if val in last_seen and last_seen[val] >= start_idx:
                start_idx = last_seen[val] + 1
            last_seen[val] = i
            
            current_len = i - start_idx + 1
            if current_len > max_len:
                max_len = current_len
                max_start = start_idx
                
        return max_len, max_start

    best_k = 1
    best_s = 1
    
    # Strategy: Scan from high to low.
    # We will try segments of size 10^5 or 10^6.
    # Start near 10^12.
    
    segment_size = 10**5
    # We'll scan a total range of 10^7 to be safe, or until we find 200.
    # Let's try scanning the last 10^6 numbers first.
    
    # To allow for finding the start s correctly, we need to handle boundaries.
    # If a run spans two segments, we need to merge.
    # Simpler: Just scan a large enough contiguous block.
    # Let's scan [10^12 - 2*10^6, 10^12].
    
    total_scan = 2 * 10**6
    L_start = MAX_N - total_scan + 1
    
    current_L = L_start
    while current_L <= MAX_N and time.time() < deadline:
        if current_L + segment_size - 1 > MAX_N:
            seg_size = MAX_N - current_L + 1
        else:
            seg_size = segment_size
            
        phis = compute_phi_segment(current_L, seg_size)
        k, rel_s = find_max_distinct_run(phis)
        
        if k > best_k:
            best_k = k
            best_s = current_L + rel_s
            
        if best_k == MAX_K:
            break
            
        current_L += seg_size
        
    # If we didn't find 200, maybe we should try a different range?
    # But 200 is very likely in 10^6 samples near 10^12.
    # If time remains, we can scan lower ranges, but they are less likely to have longer runs.
    
    return {'start': best_s, 'length': best_k}
