import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    # Sieve primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    best_k = 0
    best_p = 0
    
    # Check primes p
    for p in primes:
        if time.time() > deadline:
            break
        if p > 10**7:
            break
            
        k = 0
        for i in range(1, 201):
            v = p * p + i
            # Check if v is p-smooth
            temp = v
            smooth = True
            for q in primes:
                if q * q > temp:
                    break
                if q > p:
                    break
                while temp % q == 0:
                    temp //= q
            if temp > 1:
                if temp > p:
                    smooth = False
                else:
                    # temp is a prime <= p
                    # We already divided by all primes <= p, so if temp > 1, it must be prime
                    # But we stopped at q*q > temp. So temp is prime.
                    # If temp <= p, it's fine.
                    pass
            
            if not smooth:
                break
            k += 1
            
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
