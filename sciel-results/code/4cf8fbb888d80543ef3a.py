import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Sieve of Eratosthenes to get primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    best_k = 0
    best_p = 2
    
    # We can search up to the limit of primes generated
    # If we have time, we could extend the sieve, but 10^6 is a good start
    for p in primes:
        if time.time() > deadline:
            break
        
        # Check p-smoothness for p^2+1 to p^2+200
        # rem[i] will hold the remaining part of p^2+i+1 after dividing by primes <= p
        k_max = 200
        rem = [0] * (k_max + 1)
        for i in range(1, k_max + 1):
            rem[i] = p * p + i
        
        # Sieve with primes q <= p
        # Optimization: only iterate primes q <= p
        # We can find the index of the last prime <= p
        # Since primes are sorted, we can just iterate and break if q > p
        for q in primes:
            if q > p:
                break
            
            # Find first multiple of q in [p^2+1, p^2+k_max]
            start_val = p * p + 1
            # first multiple >= start_val
            # m = ceil((p^2+1)/q) * q
            # index i = m - p^2
            # i ranges from 1 to k_max
            
            # Calculate first i
            # p^2 + i >= q * k
            # i >= q * k - p^2
            # We want smallest i in [1, k_max] such that (p^2+i) % q == 0
            
            # (p^2 + i) % q == 0 => i % q == (-p^2) % q
            # Let r = (-p^2) % q
            # If r == 0, i can be q, 2q, ...
            # If r != 0, i can be r, r+q, ...
            # But i must be >= 1.
            
            r = (-p * p) % q
            if r == 0:
                first_i = q
            else:
                first_i = r
            
            if first_i > k_max:
                continue
                
            for i in range(first_i, k_max + 1, q):
                # Divide rem[i] by q as much as possible
                while rem[i] % q == 0:
                    rem[i] //= q
        
        # Check how many are smooth (rem[i] == 1)
        current_k = 0
        for i in range(1, k_max + 1):
            if rem[i] == 1:
                current_k += 1
            else:
                break
        
        if current_k > best_k:
            best_k = current_k
            best_p = p
            # Optional: early exit if we reach 200
            if best_k == 200:
                break

    return {'p': best_p, 'k': best_k}
