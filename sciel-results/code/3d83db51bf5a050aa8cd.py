import math
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # safety margin before 120s limit

    # Sieve of Eratosthenes
    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Precompute primes up to 2,000,000
    # This allows us to fully factorize numbers up to (2*10^6)^2 = 4*10^12
    # So we can check p up to 2*10^6 efficiently
    SIEVE_LIMIT = 2000000
    primes = sieve_primes(SIEVE_LIMIT)
    
    # Function to check if n is p-smooth using precomputed primes
    # n is around p^2, so sqrt(n) is around p.
    # If p <= SIEVE_LIMIT, we have all primes up to sqrt(n).
    def is_p_smooth(n, p):
        m = n
        # Iterate through primes up to sqrt(m)
        # Since m changes, we check q*q <= m
        for q in primes:
            if q * q > m:
                break
            if m % q == 0:
                while m % q == 0:
                    m //= q
                if m == 1:
                    return True
                # If m < p, then remaining factors are < p
                # If m >= p, we continue, but if m is prime and > p, we return False later
        # If loop finishes, m is either 1 or prime
        if m == 1:
            return True
        # m is prime
        return m <= p

    # Check run length for a given prime p
    def check_run(p, k_max=200):
        k = 0
        p_sq = p * p
        for i in range(1, k_max + 1):
            if time.time() > DEADLINE - 2:
                break
            n = p_sq + i
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        return k

    best_k = 0
    best_p = 2

    # Search primes in increasing order
    # We can check all p <= SIEVE_LIMIT
    for p in primes:
        if time.time() > DEADLINE - 5:
            break
        # Optimization: if p is very large, k is likely small.
        # But we want to find max k.
        # Just check all.
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p
            # If we find a large k, we might want to update early?
            # But we must finish to ensure no larger k exists?
            # The task is to maximize k.
            # If we find k=10, it's likely the best.
            # But let's keep searching if time permits.

    # If we have time, we could try larger p with Pollard Rho, 
    # but trial division up to 2e6 is very fast.
    # 2e6 primes is ~150k primes.
    # Checking 150k primes * average k checks.
    # k is small (1-5).
    # 150k * 5 * (avg 100 divisions) = 7.5e6 ops. Very fast.
    
    return {'p': best_p, 'k': best_k}
