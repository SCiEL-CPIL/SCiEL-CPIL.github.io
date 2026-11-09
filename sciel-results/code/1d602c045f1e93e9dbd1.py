"""
Combined approach for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9).

Strategy:
1. Precompute a sieve of small primes up to 1e6.
2. For each candidate prime p, perform a "smooth run" check for j=1..200:
   - Trial divide p^2+j by small primes up to min(p, sqrt(p^2+j)).
   - If the remaining cofactor is 1 or <= p, the number is p-smooth.
   - If the remaining cofactor is > p, it is a large prime or product of large primes.
     - If it is prime and > p, the run breaks.
     - If it is composite, we need to check if it has any factor > p. 
       However, for p up to 1e9, p^2+j is ~1e18. A composite cofactor > p 
       that is p-smooth must have factors <= p. But if we've already divided by 
       all primes <= p (or up to sqrt(n) which is ~p), the cofactor is prime.
       Wait, trial division up to sqrt(n) is too slow for n~1e18 (sqrt is 1e9).
       
   Correction on Smoothness Check:
   We need to check if ALL prime factors of N are <= p.
   If we trial divide by primes up to P_limit, and the remaining cofactor R is:
   - 1: Smooth.
   - Prime: Smooth iff R <= p.
   - Composite: 
     If R is composite, it has a factor <= sqrt(R).
     If sqrt(R) <= P_limit, we would have found that factor.
     If sqrt(R) > P_limit, R could be a product of two primes both > P_limit.
     If P_limit is small (e.g. 10^4), R could be p1*p2 where p1,p2 > 10^4.
     If p1 > p, then not smooth.
     
   For p up to 1e9, we cannot trial divide up to p.
   However, notice that for small p (e.g. p < 10^4), we can trial divide up to p easily.
   For large p, p^2 + j is roughly p^2. The prime factors are likely small or large.
   If a factor is > p, it must be a prime (since (p+1)^2 > p^2+j for j<2p+1, and 200 < 2p+1 for p>100).
   Actually, if N = p^2 + j, and N has a factor f > p, then N/f < p^2/p = p.
   So N/f is a factor < p.
   Thus, if N is not p-smooth, it has a prime factor q > p.
   Then N = q * r, with r < p.
   So r must be a divisor of N that is < p.
   This means that to check if N is p-smooth, we can check if N has any prime factor q > p.
   Equivalently, N is p-smooth iff N has no prime factor > p.
   Since any prime factor > p must be paired with a factor < p (because N ~ p^2),
   we can find such a factor by looking for divisors of N in the range [1, p].
   But checking all divisors is hard.
   
   Alternative:
   If N has a prime factor q > p, then q is prime and q | N.
   Since N < p^2 + 200, and q > p, q can be at most N.
   But q must be a prime divisor of N.
   If we can factor N partially...
   
   Let's use the property: N is p-smooth iff for all primes q > p, q does not divide N.
   This is hard to check directly.
   
   Better approach for large p:
   Compute R = N after trial division by small primes (up to 10^6).
   If R == 1, smooth.
   If R is prime: smooth iff R <= p.
   If R is composite:
     R must be a product of primes.
     If all prime factors of R are <= p, then R is p-smooth.
     If any prime factor of R is > p, then R is not p-smooth.
     Since we divided by primes up to 10^6, the smallest prime factor of R is > 10^6.
     If p < 10^6, then any prime factor of R is > p, so R is not p-smooth unless R=1.
     Wait, if p < 10^6, we should have divided by all primes up to p.
     So if p < 10^6, we can fully determine smoothness by trial division up to p.
     
     If p >= 10^6:
     R has no prime factors <= 10^6.
     If R is prime, it is > 10^6. It is <= p iff R <= p.
     If R is composite, it has at least two prime factors, both > 10^6.
     The smallest such factor is > 10^6.
     If p is large (e.g. 10^9), R could be a product of two primes ~10^9, which are <= p.
     Or R could be a prime > p.
     
     So for p >= 10^6:
     1. Trial divide by primes <= 10^6.
     2. Let R be the remainder.
     3. If R == 1: Smooth.
     4. If R is prime: Smooth iff R <= p.
     5. If R is composite:
        - We need to check if R has any prime factor > p.
        - Since R < (10^9)^2 = 10^18, R can be factored using Pollard-Brent.
        - Factor R completely. Check if all factors <= p.
        
   This seems feasible. Pollard-Brent is fast for 64-bit integers.
   The bottleneck is doing this for 200 numbers for each candidate p.
   If p is large, trial division by 10^6 primes is ~78,000 divisions.
   200 * 78,000 = 15.6 million divisions per p.
   This is too slow for many p.
   
   Optimization:
   We only need to check the run length k.
   We can stop early if a number is not smooth.
   For small p, the run length is likely short.
   For large p, the run length might be longer, but the density of p-smooth numbers might be lower?
   Actually, the probability that a random number near p^2 is p-smooth is related to the Dickman function.
   rho(u) where u = log(N)/log(p) ~ log(p^2)/log(p) = 2.
   rho(2) ~ 0.306.
   So roughly 30% of numbers are p-smooth.
   The expected run length is 1/(1-0.306) ~ 1.44.
   So long runs are rare.
   We should search for p that yield long runs.
   
   To speed up the check:
   - Precompute a list of small primes.
   - For a given p, check j=1, 2, ...
   - For each N = p^2 + j:
     - Trial divide by small primes up to min(p, 10^6).
     - If remainder R == 1: ok.
     - If R is prime: check R <= p.
     - Else: Factor R using Pollard-Brent, check factors.
   
   To find good p:
   - Search small primes first (up to 10^6) where we can do full trial division up to p quickly.
   - Then search larger primes in chunks, using the mixed method.
   
   Key insight from inspiration 1:
   Use lru_cache for factorization? No, N changes with p.
   Use efficient Pollard-Brent.
   
   Let's refine the search strategy:
   1. Sieve primes up to 10^6.
   2. For each prime p <= 10^6:
      - Check run length using trial division up to p.
      - This is fast because p is small.
   3. For primes p > 10^6:
      - Sample primes in larger ranges.
      - For each p, check run length using trial division up to 10^6 + Pollard-Brent for the remainder.
      
   We need to be careful with the 150s limit.
   Trial division up to 10^6 takes ~78k mod operations.
   In Python, this is ~0.01s per number? No, maybe 0.1s.
   200 numbers * 0.1s = 20s per p.
   We can check ~7 p's in 150s.
   This is not enough to find a good p if the probability is low.
   
   We need a faster filter.
   Instead of full trial division, we can use a precomputed table of quadratic residues?
   Or just trial divide by very small primes (up to 10^4) to filter out most non-smooth numbers quickly.
   If N has a small prime factor, we remove it.
   If the remainder is 1, smooth.
   If the remainder is small, check directly.
   If the remainder is large, it might be prime or composite.
   If we stop trial division at 10^4, the remainder R is either 1, prime, or product of primes > 10^4.
   If R is prime, check R <= p.
   If R is composite, factor it.
   
   This reduces the trial division cost significantly.
   10^4 primes is ~1200 divisions.
   200 * 1200 = 240k divisions per p.
   This is ~0.05s per p.
   We can check ~3000 p's in 150s.
   Still, finding a long run (k>10) might be rare.
   
   Let's look at the current score 2.33.
   This implies k is small.
   To improve, we need to find p with larger k.
   Maybe search more p's, or search smarter.
   
   Structural change:
   Use a segmented sieve to find primes in large ranges, and check them.
   Or, use the fact that p^2 + j must be smooth.
   For a fixed j, the condition is that p^2 + j is p-smooth.
   This is a constraint on p.
   
   Let's stick to the search over p, but optimize the smoothness check.
   
   Optimized Smoothness Check:
   1. Trial divide by primes up to 10^5 (9592 primes).
   2. Let R be the remainder.
   3. If R == 1: return True.
   4. If R is prime: return R <= p.
   5. Else: 
      - Check if R is a perfect power? (unlikely)
      - Factor R using Pollard-Brent.
      - Check if all factors <= p.
      
   We can cache the primality of R? No, R depends on p.
   
   Let's implement this.
   
   Additional optimization:
   For p < 10^5, we can trial divide up to p directly, which is faster than Pollard-Brent.
   For p >= 10^5, use the mixed method.
"""

import time
import math
import random
from functools import lru_cache

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def _is_prime_mr(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
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

# ---------- Pollard-Brent ----------
def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    while True:
        y = random.randrange(2, n - 1)
        c = random.randrange(1, n - 1)
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
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
            if g == n:
                continue
        return g

# ---------- Factorization ----------
@lru_cache(maxsize=None)
def _factor(n):
    if n == 1:
        return ()
    if _is_prime_mr(n):
        return (n,)
    d = _pollard_brent(n)
    return _factor(d) + _factor(n // d)

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
    deadline = start_time + 140  # Safety margin

    best_k = 0
    best_p = 2

    # Precompute small primes for trial division
    SMALL_LIMIT = 100_000
    small_primes = sieve_primes(SMALL_LIMIT)

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        # Trial divide by small primes
        for q in small_primes:
            if q > p:
                break
            if q * q > n:
                break
            while n % q == 0:
                n //= q
        if n == 1:
            return True
        if n <= p:
            return True
        # n > p
        if _is_prime_mr(n):
            return n <= p
        # n is composite and > p
        # Factor n and check
        factors = _factor(n)
        return all(f <= p for f in factors)

    def check_run(p, max_k=200):
        """Return the length of the run of p-smooth numbers after p^2."""
        k = 0
        p_sq = p * p
        for j in range(1, max_k + 1):
            n = p_sq + j
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        return k

    # Phase 1: Search small primes up to 10^6
    # For these, trial division up to p is feasible if we optimize.
    # But our is_p_smooth uses small_primes up to 10^5.
    # For p < 10^5, it's exact.
    # For p > 10^5, it uses Pollard-Brent for the remainder.
    
    primes_to_check = []
    
    # Generate primes up to 1e6
    primes_1e6 = sieve_primes(1_000_000)
    
    # We can check all primes up to 1e6 quickly?
    # is_p_smooth for p < 10^5 is fast.
    # For p > 10^5, it might be slower.
    
    for p in primes_1e6:
        if time.time() > deadline:
            break
        k = check_run(p, 200)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Search larger primes
    # Use random sampling or segmented sieve
    # Let's try segmented sieve for ranges [1e6, 1e7], [1e7, 1e8], etc.
    # But checking each prime is expensive.
    # We can filter primes by checking p^2+1 first.
    
    def is_prime_mr_simple(n):
        return _is_prime_mr(n)

    # Search in chunks
    chunk_size = 1_000_000
    start = 1_000_000
    max_search_limit = 10**9
    
    while start < max_search_limit and time.time() < deadline - 10:
        end = min(start + chunk_size, max_search_limit)
        
        # Segmented sieve
        is_p = bytearray(b'\x01') * (end - start + 1)
        
        # Sieve out composites
        limit = int(math.isqrt(end)) + 1
        # We need primes up to limit
        # We already have primes up to 1e6. If limit > 1e6, we need more.
        # For now, assume start < 1e12 so limit < 1e6.
        if limit > len(small_primes) or small_primes[-1] < limit:
            # Extend primes if needed
            pass # This logic is complex, stick to start < 1e12
        
        for sp in small_primes:
            if sp * sp > end:
                break
            # Find first multiple of sp in [start, end]
            first = ((start + sp - 1) // sp) * sp
            if first == sp:
                first += sp
            for multiple in range(first, end + 1, sp):
                is_p[multiple - start] = 0
                
        for i in range(end - start + 1):
            if time.time() > deadline:
                break
            if is_p[i]:
                p = start + i
                if p < 2:
                    continue
                # Quick check: is p^2+1 smooth?
                # If not, skip.
                n1 = p * p + 1
                if is_p_smooth(n1, p):
                    k = check_run(p, 200)
                    if k > best_k:
                        best_k = k
                        best_p = p

        start = end

    return {'p': best_p, 'k': best_k}
