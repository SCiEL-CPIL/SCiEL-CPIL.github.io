import math
import time
from math import gcd

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
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
        return [i for i, v in enumerate(is_p) if v]
    
    # We need primes up to a certain limit for trial division
    # If p is up to 10^6, we need primes up to 10^6
    # Let's sieve up to 10^7 to have enough candidates and divisors
    PRIME_LIMIT = 10**7
    primes = sieve_primes(PRIME_LIMIT)
    
    best_k = 0
    best_p = 2
    
    # Function to check if N is p-smooth
    # p is the limit for prime factors
    # We use the precomputed list of primes
    def is_p_smooth(N, p, primes_list):
        if N <= 1:
            return True
        R = N
        # Only need to check primes <= p
        # Also, if q*q > R, then R is prime (or 1).
        # If R > 1 and R is prime, we need to check if R <= p.
        # But we are iterating q up to p.
        
        # Optimization: if p is small, iterate all primes <= p.
        # If p is large, we might stop early if R becomes 1.
        
        for q in primes_list:
            if q > p:
                break
            if q * q > R:
                # R is prime (or 1)
                if R == 1:
                    return True
                else:
                    # R is a prime factor. Check if it's <= p
                    return R <= p
            while R % q == 0:
                R //= q
            if R == 1:
                return True
        # If we exit the loop, R might be > 1
        # Since we checked all q <= p, any remaining factor in R must be > p
        # UNLESS R == 1
        return R == 1

    # Iterate through candidate primes p
    # We will scan primes up to 10^6 first, as they are faster to check
    # and have higher density of "smooth" numbers relative to search space?
    # Actually, probability is roughly constant.
    # But checking p=10^6 takes ~78k divisions.
    # Checking p=100 takes ~25 divisions.
    
    # Let's scan primes in order.
    for p in primes:
        if time.time() > deadline:
            break
        
        # Skip very large p if we have already spent too much time
        # But let's just rely on the deadline check.
        
        k = 0
        # Check up to 200
        while k < 200:
            val = p * p + k + 1
            if is_p_smooth(val, p, primes):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # If we have time, we could try random large primes, 
    # but trial division is too slow for p > 10^6.
    # For p > 10^6, we would need Pollard-Brent or a different method.
    # Given the time limit, scanning up to 10^6 is a good bet.
    # 78,498 primes.
    # Average cost?
    # Let's hope it fits.
    
    return {'p': best_p, 'k': best_k}
