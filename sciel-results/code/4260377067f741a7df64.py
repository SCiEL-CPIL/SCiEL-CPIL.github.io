"""Long runs after p^2 with no prime factor exceeding p (p <= 10^9)."""
import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin before 150s kill

    # Precompute small primes for trial division and pre-filtering
    small_primes = []
    def sieve(n):
        is_p = [True] * (n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, n+1, i):
                    is_p[j] = False
        return [i for i in range(2, n+1) if is_p[i]]
    
    small_primes = sieve(1000)
    small_prime_set = set(small_primes)
    
    # For pre-filtering, use primes up to 200
    filter_primes = [p for p in small_primes if p <= 200]
    
    # Deterministic Miller-Rabin bases for n < 3.3e24 (covers p^2+k for p<=1e9, k<=200)
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime_mr(n):
        if n < 2:
            return False
        for p in small_primes:
            if n % p == 0:
                return n == p
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
    
    def factorize(n):
        """Full factorization of n, returns list of prime factors with multiplicity."""
        if n <= 1:
            return []
        factors = []
        # Trial divide by small primes
        for p in small_primes:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime_mr(n):
            factors.append(n)
            return factors
        # Use Pollard-Brent recursively
        stack = [n]
        while stack:
            x = stack.pop()
            if x == 1:
                continue
            if is_prime_mr(x):
                factors.append(x)
            else:
                d = pollard_brent(x)
                stack.append(d)
                stack.append(x // d)
        return factors
    
    def largest_prime_factor(n):
        """Return the largest prime factor of n."""
        if n <= 1:
            return 1
        # Trial divide by small primes
        remaining = n
        for p in small_primes:
            if p * p > remaining:
                break
            while remaining % p == 0:
                remaining //= p
        if remaining == 1:
            return small_primes[-1] if small_primes else 1
        if is_prime_mr(remaining):
            return max(remaining, small_primes[-1] if small_primes else 1)
        # Need full factorization of remaining
        factors = factorize(remaining)
        if factors:
            return max(factors)
        return 1
    
    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        remaining = n
        for sp in small_primes:
            if sp > p:
                break
            if sp * sp > remaining:
                break
            while remaining % sp == 0:
                remaining //= sp
        if remaining == 1:
            return True
        if remaining <= p:
            # remaining is prime (or 1) and <= p
            return is_prime_mr(remaining)
        # remaining has a factor > p, need to check
        # If remaining is prime and > p, not smooth
        if is_prime_mr(remaining):
            return remaining <= p
        # remaining is composite, factorize
        factors = factorize(remaining)
        for f in factors:
            if f > p:
                return False
        return True
    
    def prefilter_score(p):
        """
        Score a prime p based on how many of p^2+1..p^2+200 are divisible
        by small primes (heuristic for p-smoothness).
        Higher score = more likely to have long p-smooth run.
        """
        score = 0
        p2 = p * p
        for i in range(1, 201):
            n = p2 + i
            for sp in filter_primes:
                if sp > p:
                    break
                if n % sp == 0:
                    score += 1
                    break  # Only count once per n
        return score
    
    def check_run(p, max_k=200):
        """Check how many consecutive integers p^2+1..p^2+k are p-smooth."""
        k = 0
        p2 = p * p
        for i in range(1, max_k + 1):
            if time.time() > deadline:
                break
            if is_p_smooth(p2 + i, p):
                k += 1
            else:
                break
        return k
    
    # Known good starting points - try these first
    known_candidates = [5, 7, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    best_k = 0
    best_p = 2
    
    # First pass: check known small primes quickly
    for p in known_candidates:
        if time.time() > deadline:
            break
        if not is_prime_mr(p):
            continue
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p
    
    # Generate candidate primes using pre-filtering
    # We want primes where p^2+i tends to be divisible by small primes
    
    # Strategy: for each small prime q, p^2 + i ≡ 0 (mod q) means p^2 ≡ -i (mod q)
    # For a given p, count how many i in 1..200 have p^2 + i divisible by some small prime
    
    # Precompute for efficiency: for each small prime q, which residues r mod q
    # make r^2 + i ≡ 0 (mod q) for i in 1..200?
    # Actually, for a given p, p mod q is fixed, so p^2 mod q is fixed.
    # Then p^2 + i ≡ 0 (mod q) iff i ≡ -p^2 (mod q).
    # So for each q, there's at most one residue class of i (mod q) that works.
    # Number of i in 1..200 with i ≡ a (mod q) is either floor(200/q) or ceil(200/q).
    
    # Better approach: just compute the score directly but efficiently
    # For a prime p, compute p mod q for each filter prime q, then count hits.
    
    def fast_prefilter(p):
        """Fast pre-filter: count how many i in 1..200 have p^2+i divisible by a small prime."""
        score = 0
        p2_mods = {}
        for q in filter_primes:
            if q > p:
                break
            p2_mod = (p * p) % q
            # i ≡ -p2_mod (mod q)
            target = (-p2_mod) % q
            # Count i in 1..200 with i ≡ target (mod q)
            # i = target, target+q, target+2q, ... but i >= 1
            if target == 0:
                # i = q, 2q, 3q, ...
                count = 200 // q
            else:
                # i = target, target+q, ...
                if target <= 200:
                    count = 1 + (200 - target) // q
                else:
                    count = 0
            score += count
        return score
    
    # Search over a range of primes, using pre-filter to rank candidates
    # We'll search primes up to some limit, but prioritize high-scoring ones
    
    # Phase 1: Scan primes up to 10000, rank by prefilter, check top ones
    search_limit = 10000
    candidates = []
    for p in range(3, search_limit + 1, 2):
        if time.time() > deadline:
            break
        if p % 2 == 0:
            continue
        # Quick primality test
        is_p = True
        for q in small_primes:
            if q * q > p:
                break
            if p % q == 0:
                is_p = False
                break
        if not is_p:
            continue
        sc = fast_prefilter(p)
        candidates.append((sc, p))
    
    # Sort by score descending, check top candidates
    candidates.sort(key=lambda x: -x[0])
    
    checked = set(known_candidates)
    for sc, p in candidates:
        if time.time() > deadline:
            break
        if p in checked:
            continue
        checked.add(p)
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p
        # Early stop if we've found a very good one
        if best_k >= 50:
            break
    
    # Phase 2: If we have time, search larger primes
    if time.time() < deadline - 10:
        # Search primes in ranges, using pre-filter
        # Focus on ranges where p is large enough that p-smoothness is plausible
        search_ranges = [
            (10000, 50000),
            (50000, 200000),
            (200000, 1000000),
            (1000000, 10000000),
            (10000000, 100000000),
        ]
        
        for lo, hi in search_ranges:
            if time.time() > deadline - 5:
                break
            # Sample primes in this range
            step = max(1, (hi - lo) // 500)  # Check at most ~500 candidates per range
            for p in range(lo, hi + 1, step):
                if time.time() > deadline - 5:
                    break
                if p % 2 == 0:
                    continue
                # Quick primality
                is_p = True
                for q in small_primes:
                    if q * q > p:
                        break
                    if p % q == 0:
                        is_p = False
                        break
                if not is_p:
                    continue
                sc = fast_prefilter(p)
                # Only do full check if score is promising
                if sc > 100:  # Threshold: need decent coverage by small primes
                    k = check_run(p)
                    if k > best_k:
                        best_k = k
                        best_p = p
                    if best_k >= 80:
                        break
            if best_k >= 80:
                break
    
    # Final: verify our best answer
    if best_k > 0:
        # Double-check the run
        verified_k = 0
        p2 = best_p * best_p
        for i in range(1, best_k + 1):
            if is_p_smooth(p2 + i, best_p):
                verified_k += 1
            else:
                break
        best_k = verified_k
    
    if best_k < 1:
        # Fallback: return a valid small answer
        best_p = 2
        best_k = 1  # 2^2+1 = 5, factor 5 > 2, so not smooth... 
        # Actually for p=2, p^2+1=5, 5>2, so k=0. Need p>=3.
        # For p=3, p^2+1=10=2*5, 5>3, so k=0.
        # For p=5, p^2+1=26=2*13, 13>5, k=0.
        # For p=7, p^2+1=50=2*5^2, all factors <=7. k>=1.
        best_p = 7
        best_k = 1
    
    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
