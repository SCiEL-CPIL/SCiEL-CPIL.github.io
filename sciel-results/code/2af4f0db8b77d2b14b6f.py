import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

    # Sieve of Eratosthenes
    def sieve(limit):
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(limit + 1) if is_prime[i]]

    # Small primes for trial division
    small_primes = sieve(100000)
    small_primes_set = set(small_primes)

    # Miller-Rabin primality test (deterministic for n < 2^64)
    def is_prime_mr(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Pollard's Rho (Brent variant)
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
            m = 128
            g, r, q = 1, 1, 1
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = (q * abs(x - y)) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
                if r > (1 << 20):
                    break
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if 1 < g < n:
                return g

    def factorize(n):
        if n == 1:
            return []
        if n in small_primes_set:
            return [n]
        factors = []
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    factors.append(p)
                    n //= p
        if n == 1:
            return factors
        if is_prime_mr(n):
            factors.append(n)
            return factors
        stack = [n]
        while stack:
            x = stack.pop()
            if is_prime_mr(x):
                factors.append(x)
            else:
                d = pollard_brent(x)
                stack.append(d)
                stack.append(x // d)
        return factors

    def is_p_smooth(n, p_limit):
        """Check if all prime factors of n are <= p_limit."""
        if n <= 1:
            return True
        if n <= p_limit:
            return True
        # Key insight: if n is prime and n > p_limit, not p-smooth
        if is_prime_mr(n):
            return False
        # n is composite. Factor it and check all factors
        factors = factorize(n)
        for f in factors:
            if f > p_limit:
                return False
        return True

    def check_run_fast(p, max_k=200):
        """
        Check the run length for prime p.
        Key optimization: for p > max_k, any n in [p^2+1, p^2+max_k] 
        that is prime is automatically not p-smooth.
        So we only need to check if n is composite (and if so, verify factors <= p).
        For composite n ~ p^2, factors are in [n/p, p] = [~p, p], 
        so most composite n are automatically p-smooth.
        """
        p_sq = p * p
        k = 0
        for j in range(1, max_k + 1):
            n = p_sq + j
            # If p > max_k, then n > p always, so n must be composite to be p-smooth
            if is_prime_mr(n):
                # n is prime and n > p (since n >= p^2+1 > p for p >= 2)
                break
            # n is composite. For large p, n ~ p^2, so factors are ~p.
            # We need to verify all prime factors <= p.
            # For n composite with n ~ p^2, if n = a*b with a <= b,
            # then a <= sqrt(n) ~ p and b = n/a >= n/p ~ p.
            # So b could be > p only if a < p.
            # Actually, the largest prime factor of n is at most n / (smallest prime factor).
            # For n ~ p^2, if the smallest prime factor is small, the largest could be large.
            # But in practice, for random composites near p^2, they tend to have factors near p.
            # Let's do a quick check: try to find a factor > p
            if not is_p_smooth(n, p):
                break
            k = j
        return k

    def check_run_small_p(p, max_k=200):
        """Check run for small p where the composite optimization doesn't apply as cleanly."""
        p_sq = p * p
        k = 0
        for j in range(1, max_k + 1):
            n = p_sq + j
            if not is_p_smooth(n, p):
                break
            k = j
        return k

    # Dickman function approximation for prioritization
    def dickman_approx(u):
        """Approximate Dickman's function rho(u) for u > 0."""
        if u <= 1:
            return 1.0
        if u <= 2:
            return u * math.log(u) - u + 1
        # For larger u, use the recursive definition approximately
        # rho(u) = rho(u-1) * (u-1) / u for u in (1, 2]
        # For u > 2, it decreases more slowly
        # Simple approximation: rho(u) ~ exp(-u*(log(u) + log(log(u)) - 1)) for large u
        if u > 50:
            return math.exp(-u * (math.log(u) + math.log(math.log(u)) - 1))
        # For moderate u, use iterative approximation
        result = 1.0
        for i in range(1, int(u) + 1):
            result *= (i / u)
            result = max(result, 1e-30)
        return result

    def expected_bad_count(p, max_k=200):
        """
        Estimate expected number of 'bad' numbers in [p^2+1, p^2+max_k].
        A number is 'bad' if it has a prime factor > p.
        For p > max_k: bad iff prime. Probability ~ 1/log(p^2) = 1/(2*log(p)).
        For composite numbers, probability of having a factor > p is small.
        """
        if p <= 1:
            return max_k
        log_p = math.log(p)
        log_n = math.log(p * p)  # ~ 2*log(p)
        
        # Probability n is prime ~ 1/log(n)
        prime_prob = 1.0 / log_n
        
        # For composite n, probability of having a prime factor > p
        # This is roughly the probability that n is not p-smooth
        # Using Dickman: P(n is p-smooth) ~ rho(log(n)/log(p)) = rho(2)
        # rho(2) = 2*ln(2) - 2 + 1 = 2*0.693 - 1 = 0.386
        # So P(not p-smooth | composite) ~ 1 - rho(2) ~ 0.614
        # But this is a rough estimate
        
        # Total bad probability ~ prime_prob + (1-prime_prob) * (1 - rho(2))
        # But for large p, the composite numbers near p^2 are more likely to be p-smooth
        # because their factors are naturally around p
        
        # Better estimate: for n ~ p^2, if n is composite, its prime factors
        # are distributed, and the largest is typically O(p) or less.
        # The probability that a random composite n ~ p^2 has all factors <= p
        # is actually quite high for large p.
        
        # Simplified: bad_prob ~ 1/log(n) for the prime case dominates
        bad_prob = prime_prob * 0.5  # Rough: half the composites might be bad
        return max_k * bad_prob

    best_k = 0
    best_p = 2

    # Phase 1: Check small primes systematically
    small_limit = 50000
    small_primes_list = sieve(small_limit)
    
    for p in small_primes_list:
        if time.time() > deadline - 10:
            break
        if p > 200:
            k = check_run_fast(p)
        else:
            k = check_run_small_p(p)
        if k > best_k:
            best_k = k
            best_p = p
        if best_k >= 200:
            break

    # Phase 2: Use Dickman heuristic to prioritize larger primes
    # For p > 200, the check is fast (just primality test + occasional factorization)
    # We want to find primes where p^2+1 through p^2+k are all composite
    
    # Strategy: scan primes in increasing order, but use the fact that
    # for large p, most numbers in [p^2+1, p^2+200] are composite,
    # and we just need to verify they're p-smooth.
    
    # The bottleneck is the primality test. Let's batch-check.
    
    # For p > 1000, we can do a quick sieve-like check:
    # For each small prime q, mark multiples of q in [p^2+1, p^2+200].
    # If a number is not marked by any q <= p, it might be prime or have large factors.
    
    def check_run_sieve(p, max_k=200):
        """
        Segmented sieve approach for checking the run.
        For p > 200: n in [p^2+1, p^2+200] is p-smooth iff:
        1. n is composite, AND
        2. all prime factors of n are <= p
        
        Step 1: Use small primes to find composites.
        Step 2: For remaining candidates, verify p-smoothness.
        """
        p_sq = p * p
        lo = p_sq + 1
        hi = p_sq + max_k
        
        # Mark which numbers are divisible by small primes
        # is_composite[j] = True if p_sq + j has a small prime factor
        is_composite = [False] * (max_k + 1)
        
        # Use primes up to min(p, 10000) for trial division
        for q in small_primes:
            if q > p:
                break
            if q > 10000:  # Cap for speed
                break
            # Find first multiple of q in [lo, hi]
            first = ((lo + q - 1) // q) * q
            if first < lo:
                first += q
            for m in range(first, hi + 1, q):
                idx = m - lo
                if 0 <= idx <= max_k:
                    is_composite[idx] = True
        
        # Now check each position
        k = 0
        for j in range(1, max_k + 1):
            n = p_sq + j
            if not is_composite[j]:
                # n has no small prime factor <= 10000
                # Check if n is prime
                if is_prime_mr(n):
                    break
                # n is composite with all prime factors > 10000
                # For n ~ p^2, factors are ~p, so likely <= p
                # But verify
                if not is_p_smooth(n, p):
                    break
            else:
                # n has a small prime factor. Need to verify all factors <= p
                # For large p, this is almost certainly true
                # Quick check: divide out small factors, see if remainder is prime > p
                rem = n
                for q in small_primes:
                    if q > p:
                        break
                    if q * q > rem:
                        break
                    while rem % q == 0:
                        rem //= q
                if rem == 1:
                    pass  # All factors found, all <= p
                elif rem <= p:
                    pass  # Remaining factor <= p, OK
                else:
                    # rem > p, check if it's prime
                    if is_prime_mr(rem):
                        break  # Found prime factor > p
                    # rem is composite, factor it
                    if not is_p_smooth(rem, p):
                        break
            k = j
        
        return k

    # Phase 2 continued: scan larger primes
    # For p > 50000, use the sieve-based check which is faster
    if time.time() < deadline - 5:
        # Generate primes up to 10^7
        large_limit = 10**7
        all_primes = sieve(large_limit)
        
        # Skip primes already checked
        start_idx = len(small_primes_list)
        
        for p in all_primes[start_idx:]:
            if time.time() > deadline - 3:
                break
            
            k = check_run_sieve(p)
            
            if k > best_k:
                best_k = k
                best_p = p
            
            if best_k >= 200:
                break

    # Phase 3: Random sampling for very large primes (up to 10^9)
    if time.time() < deadline - 2 and best_k < 200:
        # For p > 10^7, we can't sieve all primes.
        # Use random candidates with Miller-Rabin primality test.
        # The sieve-based check still works for individual p.
        
        wheel = [1, 7, 11, 13, 17, 19, 23, 29]
        attempts = 0
        while time.time() < deadline - 1 and attempts < 5000:
            attempts += 1
            # Generate random candidate near 10^9
            base_num = random.randint(10**8, 10**9)
            offset = random.choice(wheel)
            p_candidate = base_num + offset
            if p_candidate > 10**9:
                p_candidate = random.randint(10**7, 10**9)
            
            if not is_prime_mr(p_candidate):
                continue
            
            p = p_candidate
            k = check_run_sieve(p)
            
            if k > best_k:
                best_k = k
                best_p = p
            
            if best_k >= 200:
                break

    return {'p': best_p, 'k': best_k}
