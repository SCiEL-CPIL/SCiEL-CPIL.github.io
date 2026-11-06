import math
import random
import time

def construct():
    # Primes up to sqrt(10^12) = 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    def compute_phis(start, length):
        """
        Computes phi for numbers start, start+1, ..., start+length-1
        using a segmented sieve approach for factorization.
        """
        if start == 1:
            # Handle 1 specially as segmented sieve usually starts >= 2
            phis = [1] * length
            if length > 1:
                # For n > 1
                for i in range(1, length):
                    n = start + i
                    m = n
                    res = n
                    for p in primes:
                        if p * p > m:
                            break
                        if m % p == 0:
                            res -= res // p
                            while m % p == 0:
                                m //= p
                    if m > 1:
                        res -= res // m
                    phis[i] = res
            return phis

        # Initialize arrays for the segment
        # rem[i] stores the remaining part of the number after dividing by small primes
        # phi_val[i] stores the current value of phi
        rem = [start + i for i in range(length)]
        phi_val = [start + i for i in range(length)]
        
        # Sieve with primes up to sqrt(start + length)
        # Actually, we need primes up to sqrt(start + length - 1)
        # But we have primes up to 10^6, which is sqrt(10^12).
        # So we can iterate through all primes.
        
        for p in primes:
            if p * p > start + length:
                break
                
            # Find the first multiple of p in the range [start, start+length-1]
            # k = ceil(start / p)
            k = (start + p - 1) // p
            if k < 1: # Should not happen for start >= 2
                k = 1
            first = k * p
            
            # Mark multiples
            for j in range(first, start + length, p):
                idx = j - start
                if rem[idx] % p == 0:
                    phi_val[idx] -= phi_val[idx] // p
                    while rem[idx] % p == 0:
                        rem[idx] //= p
        
        # Handle remaining prime factors
        for i in range(length):
            if rem[i] > 1:
                phi_val[i] -= phi_val[i] // rem[i]
                
        return phi_val

    best_k = 1
    best_s = 1
    
    start_time = time.time()
    time_limit = 115
    
    # Random search with batched evaluation
    while time.time() - start_time < time_limit:
        log_s = random.uniform(1, 12)
        s = int(10 ** log_s)
        
        if s < 1 or s + 200 > 10**12:
            continue
            
        # Evaluate batch of 200
        # We only need to check up to 200, but we can stop early if we find a collision
        # However, computing all 200 phis is fast. Let's compute all 200.
        # If s is small, we might need to handle 1.
        
        phis = compute_phis(s, 200)
        
        # Check for distinctness
        seen = set()
        k = 0
        for v in phis:
            if v in seen:
                break
            seen.add(v)
            k += 1
            
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break

    return {'start': best_s, 'length': best_k}
