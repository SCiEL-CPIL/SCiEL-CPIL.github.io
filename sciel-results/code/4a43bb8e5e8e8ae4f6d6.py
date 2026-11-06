import time
import math

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    return [i for i, v in enumerate(is_prime) if v]

def construct():
    start_time = time.time()
    deadline = start_time + 118
    
    # Precompute primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    best_start = 1
    best_len = 1
    
    # We will scan segments of size SEG_SIZE
    SEG_SIZE = 10**6
    
    # Function to compute phi for a range [L, R]
    def compute_phi_range(L, R):
        n = R - L + 1
        phi = list(range(L, R + 1))
        for p in primes:
            if p > R:
                break
            # Find first multiple of p in [L, R]
            first = ((L + p - 1) // p) * p
            for j in range(first, R + 1, p):
                idx = j - L
                phi[idx] -= phi[idx] // p
        return phi
    
    # Sliding window to find longest run of distinct values
    def find_longest_run(phi_vals, L):
        n = len(phi_vals)
        if n < 1:
            return 1, L
        max_len = 1
        max_start = L
        left = 0
        counts = {}
        
        for right in range(n):
            val = phi_vals[right]
            if val in counts:
                counts[val] += 1
            else:
                counts[val] = 1
            
            # Shrink window from left until no duplicates
            while counts[val] > 1:
                left_val = phi_vals[left]
                counts[left_val] -= 1
                left += 1
            
            # Now window [left, right] has all distinct values
            curr_len = right - left + 1
            if curr_len > max_len:
                max_len = curr_len
                max_start = L + left
        
        return max_len, max_start

    # Scan segments
    L = 1
    while L < 10**12 and time.time() < deadline:
        R = min(L + SEG_SIZE - 1, 10**12)
        phi_vals = compute_phi_range(L, R)
        k, s = find_longest_run(phi_vals, L)
        if k > best_len:
            best_len = k
            best_start = s
        L = R + 1
        
    return {'start': best_start, 'length': best_len}
