"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd, isqrt

DEADLINE = 110.0  # safety margin below 120s evaluator limit

def _sieve_primes(limit):
    """Return list of all primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _miller_rabin(n, a_list):
    """Deterministic Miller-Rabin for n < 3.3e24 using given witnesses."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in a_list:
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

def _is_prime(n):
    """Fast primality test."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Deterministic for n < 3,317,044,064,679,887,385,961,981
    return _miller_rabin(n, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])

def _pollard_brent(n, rand_func=None):
    """Pollard's rho with Brent's improvement."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if rand_func is None:
        rand_func = random.randrange
    y = rand_func(1, n - 1)
    c = rand_func(1, n - 1)
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
    if g == n:
        return None
    return g

def _is_p_smooth(val, p, small_primes):
    """
    Check if val has all prime factors <= p.
    Returns True if p-smooth, False otherwise.
    """
    if val <= 1:
        return True
    # Trial division by small primes up to p
    for sp in small_primes:
        if sp > p:
            break
        if val % sp == 0:
            while val % sp == 0:
                val //= sp
    if val == 1:
        return True
    # Check if remaining val is prime
    if _is_prime(val):
        return val <= p
    # Factor val using Pollard-Brent
    stack = [val]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if _is_prime(m):
            if m > p:
                return False
            continue
        d = None
        for attempt in range(20):
            d = _pollard_brent(m)
            if d is not None and 1 < d < m:
                break
        if d is None or d == m:
            if m > p:
                return False
        else:
            if d > p:
                return False
            if m // d > p:
                return False
            stack.append(d)
            stack.append(m // d)
    return True

def _segmented_sieve_smooth_check(p, small_primes, max_k=200, deadline=None):
    """
    Use segmented sieve to quickly determine how many consecutive integers
    p^2+1, ..., p^2+k have all prime factors <= p.
    
    Strategy:
    1. For each small prime q <= p, mark multiples in [p^2+1, p^2+max_k]
    2. After sieving, any number that becomes 1 is definitely p-smooth
    3. For numbers > 1 remaining, check if they're prime (and <= p) or factor them
    4. Return the longest consecutive run from position 1
    """
    if deadline is not None and time.time() > deadline:
        return 0
    
    p2 = p * p
    # Sieve array: remaining[i] = (p^2 + i) after dividing by small primes
    remaining = [p2 + i for i in range(max_k + 1)]  # index 0 unused
    # Track which positions are "definitely smooth" (reduced to 1)
    definitely_smooth = [False] * (max_k + 1)
    definitely_smooth[0] = True
    
    # Sieve with small primes up to min(p, some_limit)
    # We only need primes up to p, but for efficiency, use primes up to a reasonable bound
    # Numbers up to p^2 + 200 ≈ 10^18, so we need primes up to ~10^9 for full coverage
    # But we can use primes up to 10^6 and then handle the rest
    
    for q in small_primes:
        if q > p:
            break
        if deadline is not None and time.time() > deadline:
            break
        # Find the first multiple of q in [p^2+1, p^2+max_k]
        start = ((p2 + 1 + q - 1) // q) * q
        if start < p2 + 1:
            start += q
        for val in range(start, p2 + max_k + 1, q):
            idx = val - p2
            if idx < 1 or idx > max_k:
                continue
            if remaining[idx] % q == 0:
                while remaining[idx] % q == 0:
                    remaining[idx] //= q
                if remaining[idx] == 1:
                    definitely_smooth[idx] = True
    
    # Now check consecutive run from position 1
    k = 0
    for i in range(1, max_k + 1):
        if definitely_smooth[i]:
            k += 1
            continue
        # remaining[i] > 1, need to verify it's p-smooth
        rem = remaining[i]
        if rem <= 1:
            k += 1
            continue
        # Check if rem is prime and <= p
        if _is_prime(rem):
            if rem <= p:
                k += 1
                continue
            else:
                break
        # rem is composite and > 1, need to factor it
        # If any factor > p, break
        is_smooth = True
        stack = [rem]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if _is_prime(m):
                if m > p:
                    is_smooth = False
                    break
                continue
            d = None
            for attempt in range(10):
                d = _pollard_brent(m)
                if d is not None and 1 < d < m:
                    break
            if d is None or d == m:
                if m > p:
                    is_smooth = False
                    break
            else:
                if d > p or m // d > p:
                    is_smooth = False
                    break
                stack.append(d)
                stack.append(m // d)
            if deadline is not None and time.time() > deadline:
                break
        if not is_smooth:
            break
        k += 1
        if deadline is not None and time.time() > deadline:
            break
    
    return k

def _check_run_fast(p, small_primes, max_k=200, deadline=None):
    """
    Check how many consecutive integers p^2+1, ..., p^2+k have all prime factors <= p.
    Uses segmented sieve approach for speed.
    """
    return _segmented_sieve_smooth_check(p, small_primes, max_k=max_k, deadline=deadline)

def _generate_candidates_by_congruence(small_primes, max_p=10**9, num_candidates=5000):
    """
    Generate candidate primes p by solving p^2 ≡ -i (mod q) for small primes q.
    This makes p^2 + i divisible by q, increasing the chance p^2+i is p-smooth.
    """
    candidates = set()
    # Use first few small primes for congruence constraints
    constraint_primes = [q for q in small_primes if q <= 1000][:20]
    
    rng = random.Random(12345)
    
    for _ in range(num_candidates):
        if time.time() > DEADLINE - 10:
            break
        # Pick a random small prime q and offset i
        q = rng.choice(constraint_primes)
        i = rng.randint(1, 50)
        
        # Solve p^2 ≡ -i (mod q)
        # This has solutions iff -i is a quadratic residue mod q
        # Try all possible residues
        found = False
        for a in range(q):
            if (a * a) % q == (-i) % q:
                # p ≡ a (mod q) or p ≡ q-a (mod q)
                # Generate random p ≡ a (mod q) in [1000, max_p]
                min_p = 1000
                max_p = 10**9
                if max_p <= min_p:
                    break
                # p = a + k*q, find range of k
                k_min = (min_p - a + q - 1) // q
                if k_min < 0:
                    k_min = 0
                k_max = (max_p - a) // q
                if k_max > k_min:
                    k_val = rng.randint(k_min, k_max)
                    p_candidate = a + k_val * q
                    if min_p <= p_candidate <= max_p:
                        candidates.add(p_candidate)
                        found = True
                        break
                if q - a != a:  # avoid duplicate
                    k_min = (min_p - (q - a) + q - 1) // q
                    if k_min < 0:
                        k_min = 0
                    k_max = (max_p - (q - a)) // q
                    if k_max > k_min:
                        k_val = rng.randint(k_min, k_max)
                        p_candidate = (q - a) + k_val * q
                        if min_p <= p_candidate <= max_p:
                            candidates.add(p_candidate)
                            found = True
                            break
        # Also add some purely random candidates
        if rng.random() < 0.3:
            p_rand = rng.randint(1000, 10**9)
            candidates.add(p_rand)
    
    return list(candidates)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5s safety margin
    
    # Precompute small primes for trial division
    # Use primes up to 10^6 for the segmented sieve
    small_primes = _sieve_primes(1_000_000)
    
    best_k = 0
    best_p = 7  # Fallback
    
    # Phase 1: Check small primes quickly for baseline
    for p in [7, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if time.time() > deadline:
            break
        k = _check_run_fast(p, small_primes, max_k=200, deadline=deadline)
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Generate targeted candidates using congruence-based approach
    candidates = _generate_candidates_by_congruence(small_primes, max_p=10**9, num_candidates=3000)
    
    # Filter to actual primes
    prime_candidates = []
    for c in candidates:
        if 1000 <= c <= 10**9 and _is_prime(c):
            prime_candidates.append(c)
    
    # Sort by a heuristic: prefer primes where p^2+1 is likely smooth
    def _quick_score(p):
        p2 = p * p
        score = 0
        for i in range(1, 11):
            val = p2 + i
            # Count small prime divisors
            for sp in small_primes[:100]:  # first 100 primes
                if sp > p:
                    break
                if val % sp == 0:
                    score += 1
                    break
        return score
    
    prime_candidates.sort(key=_quick_score, reverse=True)
    
    # Check top candidates with the fast segmented sieve method
    for p in prime_candidates:
        if time.time() > deadline:
            break
        k = _check_run_fast(p, small_primes, max_k=200, deadline=deadline)
        if k > best_k:
            best_k = k
            best_p = p
            # If we found a good result, try nearby primes
            if best_k >= 5 and time.time() < deadline - 30:
                for delta in range(-20, 21):
                    if delta == 0:
                        continue
                    np_ = p + delta
                    if 1000 <= np_ <= 10**9 and _is_prime(np_):
                        k2 = _check_run_fast(np_, small_primes, max_k=200, deadline=deadline)
                        if k2 > best_k:
                            best_k = k2
                            best_p = np_
                    if time.time() > deadline:
                        break
    
    # Phase 3: If still time, do some random sampling as backup
    if time.time() < deadline - 10:
        rng = random.Random(99999)
        for _ in range(200):
            if time.time() > deadline:
                break
            x = rng.randint(10**5, 10**9)
            # Find nearest prime
            p = None
            for d in range(0, 500):
                for candidate in [x + d, x - d]:
                    if candidate >= 1000 and candidate <= 10**9 and _is_prime(candidate):
                        p = candidate
                        break
                if p:
                    break
            if p is None:
                continue
            k = _check_run_fast(p, small_primes, max_k=200, deadline=deadline)
            if k > best_k:
                best_k = k
                best_p = p
    
    # Ensure we have a valid result
    if best_k < 1:
        best_p = 7
        best_k = 1
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
