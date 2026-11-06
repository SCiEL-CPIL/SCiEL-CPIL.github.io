import math
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145  # 5s margin for safety
    
    # Deterministic Miller-Rabin bases for n < 2^64
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        if n % 3 == 0:
            return n == 3
        
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
            
        for a in MR_BASES:
            if a % n == 0:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
            
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factorize(n):
        factors = []
        # Trial division by very small primes first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                while n % p == 0:
                    factors.append(p)
                    n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
            return factors
        
        # Pollard Rho for remaining
        stack = [n]
        while stack:
            x = stack.pop()
            if is_prime(x):
                factors.append(x)
            else:
                d = pollard_rho(x)
                stack.append(d)
                stack.append(x // d)
        return factors

    def check_run(p):
        """
        Returns the length of the run of p-smooth numbers starting from p^2+1.
        Max 200.
        """
        k = 0
        for offset in range(1, 201):
            n = p * p + offset
            # Quick check: if n is prime, it has a factor > p (itself)
            if is_prime(n):
                break
            
            # Check if all factors <= p
            factors = factorize(n)
            if max(factors) > p:
                break
            k += 1
        return k

    # Phase 1: Sieve-based search for small primes
    # We want to find primes p such that for all i in 1..K, p^2+i is p-smooth.
    # A necessary condition is that for any prime q < p, p^2+i is not 0 mod q (unless p^2+i < q, which is impossible for i>=1, p>=2).
    # Actually, if p^2+i is divisible by q, then p^2+i = m*q.
    # If q > p, then m must be 1 (since p^2+i < p^2+p < p^2+p^2 for p>1? No. p^2+200 < 2*p^2 for p > 15).
    # So if q > p and q | p^2+i, then p^2+i is prime or a product of primes > p.
    # Specifically, if p^2+i is prime, it's a fail.
    # If p^2+i = q1*q2 with q1,q2 > p, it's a fail.
    
    # Strategy:
    # 1. Generate a list of candidate primes p up to LIMIT_P.
    # 2. For each candidate p, check if p^2+i has any small prime factor q < p.
    #    Wait, if p^2+i has a factor q < p, that's GOOD for the "smoothness" part, but we need ALL factors to be <= p.
    #    The "bad" event is having a factor > p.
    #    Factors > p can only be:
    #    a) The number p^2+i itself (if prime).
    #    b) A factor q where p < q < p^2+i.
    #    
    #    Let's use the sieve idea from the prompt:
    #    "iterate through potential large prime factors q > p and sieve out primes p where p^2 \equiv -i \pmod q"
    #    This is tricky because q depends on p.
    
    # Alternative structural change:
    # Precompute "bad" residues for small primes q.
    # If we fix a set of small primes Q = {q1, q2, ...}, we can sieve primes p.
    # For a fixed q, if p^2 + i is divisible by q, then p^2 \equiv -i (mod q).
    # This doesn't immediately help with finding p-smooth numbers, because we WANT p^2+i to have small factors.
    # The problem is finding p such that p^2+i has NO large factors.
    
    # Let's look at the "bad" numbers.
    # p^2+i is bad if it has a prime factor > p.
    # Let q be a prime factor of p^2+i with q > p.
    # Then p^2+i = m*q.
    # Since i is small (<=200), p^2+i is close to p^2.
    # So q is close to p^2/m.
    
    # Let's try a different sieve:
    # For a given p, we check i=1, 2, ...
    # We can speed up the check by precomputing for a block of p's.
    
    # Let's stick to the prompt's suggestion:
    # "iterate through potential large prime factors q > p and sieve out primes p where p^2 \equiv -i \pmod q"
    # This implies we assume q is fixed? No, q is a factor of p^2+i.
    # If we iterate q from 2 to some limit, and for each q, we find p such that p^2 \equiv -i (mod q).
    # If such a p exists, then p^2+i is divisible by q.
    # If q > p, this is BAD.
    # So we want to AVOID p such that p^2+i is divisible by any q > p.
    
    # This is still hard because q depends on p.
    
    # Let's try the "Sieve of Eratosthenes" style on the offsets.
    # For a fixed p, we want p^2+1, ..., p^2+k to be p-smooth.
    # This means for each i, p^2+i must be divisible by some prime <= p.
    
    # Let's try a different approach:
    # Random search with a better filter.
    # 1. Generate random primes p.
    # 2. For each p, check i=1. If p^2+1 is prime, reject.
    # 3. Check i=2. If p^2+2 is prime, reject.
    # 4. ...
    
    # The current program does this but with a sequential scan first.
    # The sequential scan is slow because it checks many p's that fail early.
    
    # New Strategy:
    # Use a "pre-sieve" to filter out p's that are likely to fail early.
    # For i=1, p^2+1 is prime for many p.
    # We can sieve p's such that p^2+1 is composite.
    # p^2+1 is composite if it has a factor q.
    # p^2 \equiv -1 (mod q).
    # This requires -1 to be a quadratic residue mod q, so q=2 or q \equiv 1 (mod 4).
    # So we can sieve p's such that p^2+1 is divisible by some small q \equiv 1 (mod 4).
    
    # Let's implement this.
    # 1. Generate primes up to LIMIT_P.
    # 2. For each prime q \equiv 1 (mod 4) up to LIMIT_Q, find roots of x^2 \equiv -1 (mod q).
    # 3. Mark p's in the sieve if p \equiv root (mod q).
    # 4. The remaining p's are candidates where p^2+1 is composite (and has a small factor).
    # 5. Repeat for i=2, 3, ...?
    #    For i=2, p^2+2. p^2 \equiv -2 (mod q). Requires -2 to be a QR mod q.
    
    # This is getting complex. Let's try a simpler structural change:
    # Parallelize the check_run? No, GIL.
    
    # Let's try the "bad factor" sieve more directly.
    # We want to find p such that p^2+i has NO prime factor > p.
    # This is equivalent to: for all prime factors q of p^2+i, q <= p.
    
    # Let's try a heuristic:
    # The probability that p^2+i is prime is ~ 1/log(p^2) = 1/(2 log p).
    # The probability that it's composite but has a large factor is also significant.
    
    # Let's try searching for p in a specific range where p is large enough that p^2+i is likely to have small factors, but not so large that factorization is slow.
    # Actually, larger p makes factorization slower.
    # Smaller p makes p^2+i smaller, so factorization is faster, but the condition "all factors <= p" is easier to satisfy?
    # No, for small p, p^2+i is likely to have a factor > p.
    # Example: p=2. 2^2+1=5 (prime, fail). 2^2+2=6 (factors 2,3. 3>2, fail).
    # p=3. 3^2+1=10 (2,5. 5>3, fail).
    # p=5. 5^2+1=26 (2,13. 13>5, fail).
    # p=7. 7^2+1=50 (2,5. OK). 7^2+2=51 (3,17. 17>7, fail).
    
    # It seems k is small for small p.
    # For large p, p^2+i is more likely to be prime or have large factors.
    # So the optimal p might be in the middle range.
    
    # Let's try a random search with a time limit, but with a better distribution.
    # Instead of uniform random, use a distribution that favors primes where p^2+1 is composite.
    
    # Let's implement the "pre-sieve" for i=1.
    # Sieve p's such that p^2+1 is divisible by a small prime q.
    
    LIMIT_P = 10**6
    LIMIT_Q = 10**5
    
    # Sieve primes up to LIMIT_P
    sieve = bytearray(b'\x01') * (LIMIT_P + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(LIMIT_P ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    # Pre-sieve for i=1
    # p^2+1 is divisible by q if p^2 \equiv -1 (mod q)
    # This requires q=2 or q \equiv 1 (mod 4)
    # We want p to be such that p^2+1 is NOT prime.
    # So we want p to be divisible by some q \equiv 1 (mod 4) such that -1 is a QR mod q.
    
    # Actually, we want to AVOID p's where p^2+1 is prime.
    # We can't easily sieve for "composite", but we can sieve for "has a small factor".
    # If p^2+1 has a small factor q, it's composite.
    # So we mark p's that are NOT divisible by any q such that p^2 \equiv -1 (mod q).
    # Wait, if p is NOT divisible by such q, p^2+1 might still be composite (with a large factor).
    # But if p IS divisible by such q, p^2+1 is definitely composite.
    # So we want to KEEP p's that are divisible by some small q \equiv 1 (mod 4).
    
    # Let's invert the sieve.
    # Start with all p's as "good".
    # For each q \equiv 1 (mod 4), find roots of x^2 \equiv -1 (mod q).
    # If p \equiv root (mod q), then p^2+1 is divisible by q.
    # So p is "good" for i=1.
    # If p is NOT \equiv root (mod q) for ANY small q, p might be "bad" (p^2+1 prime).
    
    # Let's try this.
    good_p = bytearray(b'\x01') * (LIMIT_P + 1)
    good_p[0] = good_p[1] = 0
    
    for q in primes:
        if q > LIMIT_Q:
            break
        if q == 2 or q % 4 == 1:
            # Find roots of x^2 \equiv -1 (mod q)
            # For q=2, roots are 1
            # For q \equiv 1 (mod 4), roots exist
            # We can find them by searching or using Tonelli-Shanks
            # For simplicity, search
            root_found = False
            for x in range(1, q):
                if (x * x) % q == q - 1:
                    # x and q-x are roots
                    # Mark p's such that p \equiv x (mod q) or p \equiv q-x (mod q)
                    # These p's are GOOD for i=1
                    # We want to keep them.
                    # So we don't mark them as bad.
                    # We want to mark p's that are NOT divisible by any such q as BAD?
                    # No, we want to find p's that ARE divisible by some such q.
                    # So we can create a "has_small_factor" array.
                    pass
            
    # This is getting complicated. Let's just use the random search but with a better filter.
    # Filter: p^2+1 must be composite.
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Small primes sequential
    for p in primes:
        if time.time() - start_time > DEADLINE - 30:
            break
        if p > 10000: # Only check small primes sequentially
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Random search with filter
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 10:
        attempts = 0
        while time.time() - start_time < DEADLINE:
            # Pick random prime in [10000, 10^6]
            # Use primes list
            idx = random.randint(0, len(primes)-1)
            p = primes[idx]
            if p < 10000 or p > 10**6:
                continue
            
            # Filter: p^2+1 must be composite
            n = p*p + 1
            if is_prime(n):
                continue
            
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p
            attempts += 1
            if attempts > 5000:
                break

    if best_k < 1:
        best_p = 7
        best_k = 1
        
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
