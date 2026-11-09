"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache

def _factorize(n):
    """Return prime factorization as list of (p, e) pairs."""
    if n <= 1:
        return []
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            factors.append((d, e))
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append((n, 1))
    return factors

@lru_cache(maxsize=1000000)
def _sigma_cached(n):
    """Compute sigma(n) = sum of divisors of n using factorization."""
    if n <= 1:
        return n
    factors = _factorize(n)
    result = 1
    for p, e in factors:
        result *= (p**(e+1) - 1) // (p - 1)
    return result

def _sigma(n):
    return _sigma_cached(n)

def _divisors(n):
    """Generate all divisors of n."""
    factors = _factorize(n)
    divs = [1]
    for p, e in factors:
        new_divs = []
        for d in divs:
            for i in range(e + 1):
                new_divs.append(d * (p ** i))
        divs = new_divs
    return divs

def _count_solutions(M):
    """Count distinct k <= sqrt(M) with k*sigma(k) = M."""
    if M < 1:
        return 0
    sqrtM = math.isqrt(M)
    solutions = set()
    # For each divisor k of M, check if k <= sqrtM and k*sigma(k) == M
    for k in _divisors(M):
        if k > sqrtM:
            continue
        if k * _sigma(k) == M:
            solutions.add(k)
    return len(solutions), solutions

def _is_highly_composite_candidate(n, max_prime=47):
    """Check if n has many small prime factors (good for many divisors)."""
    factors = _factorize(n)
    # Count total prime factors with multiplicity
    total = sum(e for _, e in factors)
    # All primes should be small for M <= 10^18
    for p, e in factors:
        if p > max_prime:
            return False
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Strategy: enumerate M values that are products of small primes
    # with high multiplicities, as these tend to have many divisors
    # and thus potentially many k*sigma(k) = M solutions.
    
    # Small primes to use
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Generate candidate M values by taking products of small primes
    # We want M to have many divisors, so use high multiplicities of small primes
    
    # Pre-compute: for a given M, the number of solutions depends on
    # how many divisors k of M satisfy k*sigma(k) = M.
    
    # Let's try M = 2^a * 3^b * 5^c * 7^d * ...
    # and search over exponents.
    
    # For efficiency, we'll focus on M with primes up to ~47
    # and total value <= 10^18
    
    # Generate candidates systematically
    candidates = set()
    
    # Add the known good one
    candidates.add(333312)
    
    # Generate M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f
    # with a >= b >= c >= d >= e >= f >= 0 and product <= 10^18
    
    max_val = 10**18
    
    # We'll do a bounded search over exponents
    # 2^a * 3^b * 5^c * 7^d * 11^e * 13^f <= 10^18
    # Max a: log2(10^18) ≈ 59.79, so a up to 59
    # But we want many divisors, so balanced exponents are better
    
    # Let's try exponents from 0 to some max for each prime
    # and collect all valid M values
    
    primes_to_use = [2, 3, 5, 7, 11, 13]
    max_exp = {2: 20, 3: 14, 5: 10, 7: 8, 11: 6, 13: 5}
    
    # Generate all combinations
    def gen_candidates(idx, current, exps):
        if time.time() > deadline - 5:
            return
        if idx == len(primes_to_use):
            if current <= max_val:
                candidates.add(current)
            return
        p = primes_to_use[idx]
        for e in range(max_exp[p] + 1):
            new_current = current * (p ** e)
            if new_current > max_val:
                break
            exps.append(e)
            gen_candidates(idx + 1, new_current, exps)
            exps.pop()
    
    gen_candidates(0, 1, [])
    
    # Also add some more structured candidates
    # M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g * 19^h
    # for additional primes
    extra_primes = [17, 19, 23, 29, 31]
    for extra_p in extra_primes:
        for base in list(candidates):
            if base * extra_p <= max_val:
                candidates.add(base * extra_p)
                if base * extra_p * extra_p <= max_val:
                    candidates.add(base * extra_p * extra_p)
    
    # Filter: only consider M with many divisors (heuristic)
    # and sort by number of divisors (descending) to find good ones first
    
    def num_divisors(n):
        factors = _factorize(n)
        result = 1
        for _, e in factors:
            result *= (e + 1)
        return result
    
    # Sort candidates by num_divisors descending
    # But this could be expensive for many candidates, so limit
    
    if len(candidates) > 50000:
        # Sample or limit
        import random
        random.seed(42)
        candidates = random.sample(list(candidates), 50000)
    
    # Sort by num_divisors
    sorted_candidates = sorted(candidates, key=lambda x: -num_divisors(x))
    
    # Evaluate top candidates
    evaluated = 0
    for M in sorted_candidates:
        if time.time() > deadline - 10:
            break
        if M < 1 or M > 10**18:
            continue
        count, solutions = _count_solutions(M)
        evaluated += 1
        if count > best_count:
            best_count = count
            best_M = M
            best_k = sorted(solutions)
            # If we find a lot, we can stop early if time is tight
            if best_count >= 50 and time.time() > deadline - 30:
                break
    
    # Ensure k values are valid
    sqrtM = math.isqrt(best_M)
    valid_k = []
    for k in best_k:
        if 1 <= k <= sqrtM and isinstance(k, int) and k * _sigma(k) == best_M:
            valid_k.append(k)
    
    if not valid_k:
        # Fallback to known good
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Ensure types are correct
    best_k = [int(k) for k in valid_k]
    
    return {'M': int(best_M), 'k': best_k}

# EVOLVE-BLOCK-END
