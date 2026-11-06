"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    def compute_phi_segment(start, end, primes, max_prime):
        """
        Compute phi(n) for n in [start, end] using a segmented sieve.
        primes: list of primes up to max_prime
        Returns array of phi values for numbers start..end (inclusive).
        """
        n = end - start + 1
        phi = np.arange(start, end + 1, dtype=np.int64)
        
        for p in primes:
            if p * p > end:
                break
            # Find the first multiple of p in [start, end]
            first = ((start + p - 1) // p) * p
            if first == p:
                first += p  # skip p itself if it's in range, handle separately
            for multiple in range(first, end + 1, p):
                idx = multiple - start
                if phi[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
        
        # Handle primes > sqrt(end) that appear in [start, end]
        # For each number in range, if it's prime, phi(p) = p-1
        # We need to identify which numbers in [start, end] are prime
        # and haven't been processed yet (i.e., they're > max_prime or their phi wasn't updated)
        
        # Actually, the above loop handles all primes up to sqrt(end).
        # For composite numbers with a prime factor > sqrt(end), that factor appears only once,
        # so the number itself must be that prime (or product of two primes both > sqrt(end), 
        # but then the number > end). So we just need to check if remaining phi[n] == n,
        # which means n is prime.
        for i in range(n):
            if phi[i] == start + i:
                # start + i is prime
                phi[i] = start + i - 1
        
        return phi
    
    def sieve_primes(limit):
        """Generate all primes up to limit using simple sieve."""
        if limit < 2:
            return []
        is_prime = np.ones(limit + 1, dtype=bool)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        primes = np.where(is_prime)[0]
        return primes.tolist()
    
    def find_longest_distinct_run_sliding(phi_vals):
        """
        Find the longest contiguous subsequence with distinct values using sliding window.
        phi_vals: numpy array of phi values
        Returns (max_length, start_index_in_phi_vals)
        """
        n = len(phi_vals)
        if n < 2:
            return 1, 0
            
        last_seen = {}
        left = 0
        max_len = 0
        max_start = 0
        
        for right in range(n):
            val = int(phi_vals[right])
            if val in last_seen and last_seen[val] >= left:
                left = last_seen[val] + 1
            last_seen[val] = right
            
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
                max_start = left
                
        return max_len, max_start
    
    best_k = 1
    best_s = 1
    
    # Generate primes up to sqrt(10^12) = 10^6 for segmented sieve
    # Actually, we only need primes up to sqrt(max_end) where max_end is our search limit
    # Let's search up to about 5*10^7 or so, so sqrt is about 7071
    # But for safety, let's generate primes up to 10^6
    
    max_search_limit = 50000000  # Search up to 50 million
    max_prime_needed = int(math.isqrt(max_search_limit)) + 1
    primes = sieve_primes(max_prime_needed)
    
    # Use segmented approach: process in chunks
    chunk_size = 5000000  # 5 million per chunk
    
    # First, do a quick scan with a smaller range to get a baseline
    # Then extend if time permits
    
    # Start with a reasonable chunk
    search_start = 1
    search_end = min(max_search_limit, chunk_size)
    
    while search_start <= max_search_limit and time_remaining() > 20.0:
        if time_remaining() < 10.0:
            break
            
        current_end = min(search_start + chunk_size - 1, max_search_limit)
        
        try:
            phi_vals = compute_phi_segment(search_start, current_end, primes, max_prime_needed)
            
            k, rel_start = find_longest_distinct_run_sliding(phi_vals)
            s = rel_start + search_start
            
            if k > 200:
                k = 200
                # If we found a run longer than 200, we can cap it
                # But we should verify the first 200 are distinct
                # For now, just cap it
            
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            break
            
        search_start = current_end + 1
    
    # Ensure valid return
    if best_k < 1:
        best_k = 1
        best_s = 1
        
    # Final safety cap
    if best_k > 200:
        best_k = 200
        
    # Verify s + k - 1 <= 10^12
    if best_s + best_k - 1 > 10**12:
        best_k = 1
        best_s = 1
        
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
