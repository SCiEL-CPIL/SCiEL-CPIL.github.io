import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
        
    def compute_phi_sieve_numpy(limit):
        """Compute phi for 0..limit using NumPy with prime-only iteration."""
        phi = np.arange(limit + 1, dtype=np.int64)
        
        # First, find all primes up to limit using a simple sieve
        is_prime = np.ones(limit + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        
        # Sieve of Eratosthenes to find primes
        for i in range(2, int(np.sqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        
        # Get list of primes
        primes = np.where(is_prime)[0]
        
        # For each prime, update phi values
        for p in primes:
            if p > limit:
                break
            indices = np.arange(p, limit + 1, p)
            phi[indices] -= phi[indices] // p
        
        return phi

    def find_longest_distinct_run(phi_vals):
        """
        phi_vals: array where phi_vals[i] = phi(i)
        Finds max length k and start s (1-indexed) such that phi(s)..phi(s+k-1) are distinct.
        """
        n = len(phi_vals)
        if n < 2:
            return 1, 1
            
        last_seen = {}
        start = 1 # 1-indexed start of current window
        max_len = 0
        max_start = 1
        
        # Iterate from 1 to n-1 (indices in phi_vals)
        # phi_vals[0] is phi(0), we ignore it for s>=1
        for i in range(1, n):
            val = phi_vals[i]
            if val in last_seen and last_seen[val] >= start:
                start = last_seen[val] + 1
            last_seen[val] = i
            
            current_len = i - start + 1
            if current_len > max_len:
                max_len = current_len
                max_start = start
                
        return max_len, max_start

    best_k = 1
    best_s = 1
    
    # Try increasing limits with more aggressive scaling
    limits = [20_000_000, 100_000_000, 300_000_000, 600_000_000]
    
    for limit in limits:
        if time_remaining() < 30.0: # Need time for sliding window too
            break
            
        try:
            phi_vals = compute_phi_sieve_numpy(limit)
            k, s = find_longest_distinct_run(phi_vals)
            
            if k > 200:
                k = 200
                
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            # Memory error or other
            break
            
    return {'start': best_s, 'length': best_k}
