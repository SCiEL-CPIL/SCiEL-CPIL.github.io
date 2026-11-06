import time
import math
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Sieve of Eratosthenes
    # 10^7 is a good balance between memory and search space coverage
    limit = 10_000_000
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    best_k = 0
    best_p = 2

    # Optimized smoothness check
    # primes_list is the global list
    # p_idx is the index such that primes_list[:p_idx] are <= p
    def is_p_smooth(n, p, primes_list, p_idx):
        m = n
        # Iterate through primes <= p
        for i in range(p_idx):
            q = primes_list[i]
            # If q^2 > m, m is prime (or 1)
            if q * q > m:
                return m <= p
            while m % q == 0:
                m //= q
            if m == 1:
                return True
        # If we exit the loop, m has no prime factors <= p
        # Since we checked all q <= p, m is either 1 or a prime > p
        # (or product of primes > p, but m < p^2, so m can only be prime > p)
        return m == 1

    # Iterate through primes
    # We can pre-compute p_idx for each p? No, bisect is fast enough.
    for p in primes:
        if time.time() > deadline:
            break
        
        # Find index of p in primes list
        # bisect_right returns index after p
        p_idx = bisect.bisect_right(primes, p)
        
        p_sq = p * p
        k = 0
        
        for i in range(1, 201):
            n = p_sq + i
            if is_p_smooth(n, p, primes, p_idx):
                k = i
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break

    return {'p': best_p, 'k': best_k}
