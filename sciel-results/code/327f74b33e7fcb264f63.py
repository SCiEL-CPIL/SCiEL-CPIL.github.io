"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from functools import lru_cache

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    start_time = time.time()
    
    # Deterministic Miller-Rabin bases for n < 3,317,044,064,679,887,385,961,981
    # For n < 3,474,749,660,383, bases 2,3,5,7,11,13,17 suffice
    # For n < 341,550,071,728,321, bases 2,3,5,7,11,13,17,19,23,29,31,37 suffice
    # Since p <= 10^9, p^2 + k <= 10^18 + 200, we need bases for n < 10^18
    # Actually for n < 3,317,044,064,679,887,385,961,981 (about 3.3e24), 
    # bases 2,3,5,7,11,13,17 are sufficient? No.
    # Let's use the standard set for 64-bit integers:
    # 2, 3, 5, 7, 11, 13, 17 is not enough for all 64-bit.
    # The set {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37} works for n < 3.3e24.
    # Since our numbers are at most ~10^18, {2, 3, 5, 7, 11, 13, 17} is actually sufficient?
    # According to literature:
    # If n < 3,474,749,660,383, then it is enough to test a = 2, 3, 5, 7, 11, 13, 17.
    # If n < 341,550,071,728,321, then it is enough to test a = 2, 3, 5, 7, 11, 13, 17, 19, 23.
    # If n < 3,825,123,056,546,413,051, then it is enough to test a = 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37.
    # Our n is at most 10^18 + 200 < 3.8e18. So we need the 12 bases.
    
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime_mr(n):
        if n < 2:
            return False
        # Small primes check
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Write n-1 as d * 2^r
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in MR_BASES:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        # Try a few random seeds
        for _ in range(20):
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = 128
            g = r = q = 1
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = q * abs(x - y) % n
                    g = gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = gcd(abs(x - ys), n)
            if g != n:
                return g
        return n  # Failed to factor
    
    @lru_cache(maxsize=None)
    def factorize(n):
        """Return list of prime factors of n."""
        if n == 1:
            return []
        if is_prime_mr(n):
            return [n]
        d = pollard_brent(n)
        if d == n:
            # Fallback: trial division up to 10^6
            for i in range(2, 10**6):
                if n % i == 0:
                    return factorize(i) + factorize(n // i)
            # If still not factored, assume prime (should not happen for our range)
            return [n]
        return factorize(d) + factorize(n // d)
    
    def max_prime_factor(n):
        """Return the largest prime factor of n."""
        if n == 1:
            return 1
        factors = factorize(n)
        return max(factors)
    
    def run_length(p):
        """Calculate the length of the run starting at p^2 + 1."""
        k = 0
        while k < 200:
            n = p * p + k + 1
            if max_prime_factor(n) > p:
                break
            k += 1
            if time.time() > deadline:
                break
        return k
    
    # Strategy:
    # 1. Try small primes first (fast to compute, might find a good run quickly)
    # 2. Then try larger primes in a smart way.
    # The key insight is that we want p^2 + i to be p-smooth for as many i as possible.
    # This is related to the distribution of smooth numbers.
    
    # Let's start with a few known good candidates or just scan.
    # The seed program found k=2 with p=3 (3^2+1=10=2*5, 3^2+2=11 is prime > 3).
    # Wait, 3^2+1 = 10. Factors 2,5. 5 > 3. So k=0 for p=3?
    # Let's check p=5. 5^2+1=26=2*13. 13>5. k=0.
    # p=7. 7^2+1=50=2*5^2. 5<=7. 7^2+2=51=3*17. 17>7. k=1.
    # p=11. 11^2+1=122=2*61. 61>11. k=0.
    # p=13. 13^2+1=170=2*5*17. 17>13. k=0.
    # p=17. 17^2+1=290=2*5*29. 29>17. k=0.
    # p=19. 19^2+1=362=2*181. 181>19. k=0.
    # p=23. 23^2+1=530=2*5*53. 53>23. k=0.
    # p=29. 29^2+1=842=2*421. 421>29. k=0.
    # p=31. 31^2+1=962=2*13*37. 37>31. k=0.
    # p=37. 37^2+1=1370=2*5*137. 137>37. k=0.
    # p=41. 41^2+1=1682=2*29^2. 29<=41. 41^2+2=1683=3^2*11*17. 17<=41. 41^2+3=1684=2^2*421. 421>41. k=2.
    # So p=41 gives k=2.
    
    # We need to search for larger k.
    # Let's generate primes up to some limit and test them.
    # Since p can be up to 10^9, we can't test all.
    # But long runs are rare. We might need to search strategically.
    
    # One idea: search for primes p such that p^2 + i is smooth for i=1..k.
    # This is a hard problem.
    
    # Let's first try a random search over a range of primes.
    # Generate primes up to 10^6 or so, and test them.
    # Then maybe try some larger primes.
    
    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    # Test primes up to 10^5 first
    primes = sieve(10**5)
    best_k = 0
    best_p = 2
    
    for p in primes:
        if time.time() > deadline:
            break
        k = run_length(p)
        if k > best_k:
            best_k = k
            best_p = p
            # If we find a large k, we might want to stop early? No, keep searching.
    
    # If we have time, try some larger primes
    if time.time() < deadline - 10:
        # Try random primes in [10^5, 10^6]
        for _ in range(1000):
            if time.time() > deadline:
                break
            p = random.randint(10**5, 10**6)
            # Find next prime
            while not is_prime_mr(p):
                p += 1
            k = run_length(p)
            if k > best_k:
                best_k = k
                best_p = p
        
        # Try random primes in [10^6, 10^7]
        if time.time() < deadline - 5:
            for _ in range(100):
                if time.time() > deadline:
                    break
                p = random.randint(10**6, 10**7)
                while not is_prime_mr(p):
                    p += 1
                k = run_length(p)
                if k > best_k:
                    best_k = k
                    best_p = p
    
    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
