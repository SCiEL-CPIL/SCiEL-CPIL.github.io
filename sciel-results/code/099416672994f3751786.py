import math
import time
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit
    LIMIT = 10**18

    def get_divisors_from_factors(factors):
        """Generate all divisors from list of (prime, exponent) pairs."""
        divisors = [1]
        for p, e in factors:
            new_divs = []
            for d in divisors:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divisors = new_divs
        divisors.sort()
        return divisors

    def tau_perp(n, factors=None):
        """Compute the score: number of adjacent coprime pairs in sorted divisors."""
        if factors is None:
            factors = []
            temp = n
            p = 2
            while p * p <= temp:
                if temp % p == 0:
                    e = 0
                    while temp % p == 0:
                        temp //= p
                        e += 1
                    factors.append((p, e))
                p += 1 if p == 2 else 2
            if temp > 1:
                factors.append((temp, 1))
        
        divisors = get_divisors_from_factors(factors)
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
        return score, len(divisors)

    best_n = 1
    best_score = 0

    # Baseline: product of first 15 primes
    primes_15 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n_base = 1
    for p in primes_15:
        n_base *= p
    if n_base <= LIMIT:
        score_base, _ = tau_perp(n_base, [(p, 1) for p in primes_15])
        best_n = n_base
        best_score = score_base

    if time.time() - start_time > DEADLINE:
        return best_n

    # Structural change: try different prime sets by swapping in larger primes
    # The idea: use primes that are more evenly distributed in log-space
    # to spread divisors more uniformly in linear space

    # Try replacing the largest few primes with slightly larger ones
    # First, generate a list of primes up to ~200
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    # Generate primes up to 200
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in range(101, 200):
        if is_prime(p):
            small_primes.append(p)

    # Strategy: fix the first 12 primes (2 through 41), vary the last 3
    # to explore different prime sets
    fixed_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    fixed_prod = 1
    for p in fixed_primes:
        fixed_prod *= p

    # Vary the last 3 primes from candidates 41..199
    candidates = [p for p in small_primes if p >= 41]
    
    # Try all combinations of 3 primes from candidates (sorted)
    # But limit to reasonable search
    import itertools
    comb_count = 0
    for combo in itertools.combinations(candidates, 3):
        if time.time() - start_time > DEADLINE:
            break
        n = fixed_prod * combo[0] * combo[1] * combo[2]
        if n > LIMIT:
            continue
        factors = [(p, 1) for p in fixed_primes] + [(combo[0], 1), (combo[1], 1), (combo[2], 1)]
        score, _ = tau_perp(n, factors)
        if score > best_score:
            best_score = score
            best_n = n
        comb_count += 1

    if time.time() - start_time > DEADLINE:
        return best_n

    # Also try 14 primes with one having exponent 2
    # n = 2^2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    primes_14 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    n_14 = 1
    for p in primes_14:
        n_14 *= p
    # Try 2^2
    n_14_sq = n_14 * 2
    if n_14_sq <= LIMIT:
        factors_14_sq = [(2, 2)] + [(p, 1) for p in primes_14[1:]]
        score_14_sq, _ = tau_perp(n_14_sq, factors_14_sq)
        if score_14_sq > best_score:
            best_score = score_14_sq
            best_n = n_14_sq

    # Try 3^2
    n_14_sq3 = n_14 * 3
    if n_14_sq3 <= LIMIT:
        factors_14_sq3 = [(2, 1), (3, 2)] + [(p, 1) for p in primes_14[2:]]
        score_14_sq3, _ = tau_perp(n_14_sq3, factors_14_sq3)
        if score_14_sq3 > best_score:
            best_score = score_14_sq3
            best_n = n_14_sq3

    # Try 5^2
    n_14_sq5 = n_14 * 5
    if n_14_sq5 <= LIMIT:
        factors_14_sq5 = [(2, 1), (3, 1), (5, 2)] + [(p, 1) for p in primes_14[3:]]
        score_14_sq5, _ = tau_perp(n_14_sq5, factors_14_sq5)
        if score_14_sq5 > best_score:
            best_score = score_14_sq5
            best_n = n_14_sq5

    return best_n
