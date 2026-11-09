import time
import random
import math
import sys

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
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
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        if g != n:
            return g

def sieve_primes(limit):
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def construct():
    start_time = time.time()
    DEADLINE = 115  # safety margin before 120s kill

    # Precompute small primes up to 10^6 for fast trial division
    SMALL_LIMIT = 10**6
    small_primes = sieve_primes(SMALL_LIMIT)
    small_primes_set = set(small_primes)
    
    # Also precompute primes up to 2*10^6 for slightly larger trial division if needed
    # But 10^6 is a good balance. Let's stick to 10^6 for memory/speed.
    
    def is_p_smooth_fast(n, p):
        """Check if all prime factors of n are <= p using precomputed primes."""
        if n <= 1:
            return True
        # Divide out small prime factors up to min(p, sqrt(n))
        # Optimization: if p >= sqrt(n), we only need to check if n is prime.
        # But we don't know sqrt(n) easily without isqrt.
        # The loop below handles it: break if q*q > n.
        for q in small_primes:
            if q > p or q * q > n:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
        if n == 1:
            return True
        # n is now prime or composite with factors > SMALL_LIMIT (or > p if p < SMALL_LIMIT)
        # If n is prime, check if n <= p
        if is_prime(n):
            return n <= p
        # Composite: factor with Pollard-Brent and recurse
        # We need to check if all factors of n are <= p
        # Since we've removed all factors <= min(p, SMALL_LIMIT), any remaining factor must be > min(p, SMALL_LIMIT)
        # If p < SMALL_LIMIT, then remaining factors are > p, so return False immediately?
        # Wait. If p < SMALL_LIMIT, the loop `if q > p` would have broken early.
        # So if p < SMALL_LIMIT, we only divided by primes <= p.
        # If n > 1, it has a prime factor > p. So return False.
        if p < SMALL_LIMIT:
            return False
        
        # If p >= SMALL_LIMIT, we divided by all primes <= SMALL_LIMIT.
        # So remaining n has no prime factors <= SMALL_LIMIT.
        # If n is composite, its smallest prime factor is > SMALL_LIMIT.
        # We need to factor n and check if all factors <= p.
        # Since smallest factor > SMALL_LIMIT, and p >= SMALL_LIMIT, we need to check against p.
        
        # Factor n completely? Or just find one factor?
        # If we find a factor f, we need to check f <= p and n//f <= p (recursively).
        # Actually, if n is composite and all its prime factors are > SMALL_LIMIT,
        # and we want to know if they are all <= p.
        
        # Let's factor n.
        # Optimization: if n is a perfect square of a prime?
        # Just use Pollard-Brent.
        
        f = pollard_brent(n)
        return is_p_smooth_fast(f, p) and is_p_smooth_fast(n // f, p)

    best_k = 0
    best_p = 2

    # Phase 1: Exhaustive search on small primes (up to 2*10^6)
    # Why 2*10^6? Because for small p, p^2 is small, and factorization is fast.
    # Also, small primes are more likely to have long runs? Or less?
    # Actually, for small p, p^2+i is small, so it has small prime factors more easily.
    # But the condition is "all prime factors <= p".
    # For small p, this is a very strong condition.
    # Example: p=3. p^2=9. 10 (2*5, 5>3 Fail). So k=0.
    # p=5. p^2=25. 26 (2*13, 13>5 Fail). k=0.
    # p=7. p^2=49. 50 (2*5*5, all <=7). 51 (3*17, 17>7 Fail). k=1.
    # p=11. p^2=121. 122 (2*61, 61>11 Fail). k=0.
    # p=13. p^2=169. 170 (2*5*17, 17>13 Fail). k=0.
    # p=17. p^2=289. 290 (2*5*29, 29>17 Fail). k=0.
    # p=19. p^2=361. 362 (2*181, 181>19 Fail). k=0.
    # p=23. p^2=529. 530 (2*5*53, 53>23 Fail). k=0.
    # p=29. p^2=841. 842 (2*421, 421>29 Fail). k=0.
    # p=31. p^2=961. 962 (2*13*37, 37>31 Fail). k=0.
    # p=37. p^2=1369. 1370 (2*5*137, 137>37 Fail). k=0.
    # p=41. p^2=1681. 1682 (2*29*29, all <=41). 1683 (3*3*11*17, all <=41). 1684 (2*2*421, 421>41 Fail). k=2.
    # It seems small primes have small k.
    # As p grows, p^2 grows, and the numbers p^2+i are larger.
    # But the bound on prime factors is p.
    # The probability that a random number N has all prime factors <= p is related to the p-smoothness of N.
    # For N ~ p^2, the probability that N is p-smooth is roughly 1/log(p) ? Or higher?
    # Dickman function. rho(u) where u = log(N)/log(p) = 2.
    # rho(2) = 1 - ln(2) ≈ 0.3.
    # So roughly 30% of numbers near p^2 are p-smooth.
    # The probability of k consecutive p-smooth numbers is (0.3)^k.
    # For k=10, (0.3)^10 ≈ 5e-6.
    # We need to search enough primes to find such a run.
    # Number of primes up to X is ~ X/ln(X).
    # If we search up to 10^9, we have ~ 50 million primes.
    # Expected number of runs of length 10: 50M * 5e-6 = 250.
    # So we should find runs of length 10 or more with high probability.
    # Runs of length 20: (0.3)^20 ≈ 3e-10. 50M * 3e-10 = 0.015. Unlikely.
    # So the max k is likely around 10-15.
    
    # Strategy:
    # 1. Search small primes exhaustively.
    # 2. Search larger primes randomly or in blocks.
    
    # Let's use the sieve primes up to 10^6 first.
    primes_1m = sieve_primes(10**6)
    
    for p in primes_1m:
        if time.time() - start_time > DEADLINE:
            break
        base = p * p
        k = 0
        while k < 200:
            if time.time() - start_time > DEADLINE:
                break
            if is_p_smooth_fast(base + k + 1, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            if k >= 200:
                break

    # Phase 2: Search primes between 10^6 and 10^9
    # We can't sieve up to 10^9.
    # We can use random sampling or segmented sieve.
    # Random sampling is simpler.
    # But random sampling might miss clusters.
    # Let's try a mix.
    
    # First, try some specific ranges or random large primes.
    # To be efficient, we should check p^2+1 first. If it fails, k=0.
    # If it passes, check p^2+2, etc.
    
    # Let's do random sampling of primes in [10^6, 10^9].
    # How to generate random primes in [L, R]?
    # Pick random odd number, check primality.
    
    num_tries = 0
    while time.time() - start_time < DEADLINE:
        num_tries += 1
        # Pick a random number in [10^6, 10^9]
        # Ensure it's odd
        candidate = random.randint(10**6, 10**9)
        if candidate % 2 == 0:
            candidate += 1
        
        # Check if prime
        if not is_prime(candidate):
            continue
        
        base = candidate * candidate
        k = 0
        while k < 200:
            if time.time() - start_time > DEADLINE:
                break
            if is_p_smooth_fast(base + k + 1, candidate):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = candidate
            if k >= 200:
                break

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    print(construct())
