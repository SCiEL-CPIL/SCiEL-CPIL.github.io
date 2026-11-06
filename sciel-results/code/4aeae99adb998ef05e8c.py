import math
import time
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit
    LIMIT = 10**18

    def elapsed():
        return time.time() - start_time

    def get_divisors_from_factors(factors):
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

    # First compute the known good baseline: 15-prime squarefree
    primes_15 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n15 = 1
    for p in primes_15:
        n15 *= p
    score15, nd15 = tau_perp(n15, [(p, 1) for p in primes_15])

    best_n = n15
    best_score = score15

    if elapsed() > DEADLINE:
        return best_n

    # Try 14 primes
    primes_14 = primes_15[:14]
    n14 = 1
    for p in primes_14:
        n14 *= p
    s14, _ = tau_perp(n14, [(p, 1) for p in primes_14])
    if s14 > best_score:
        best_n, best_score = n14, s14

    # Now explore: can mixing exponents with slightly fewer primes beat 15-prime?
    # The 15-prime has 32768 divisors. To beat score 22, we need many more coprime pairs.
    # Let's try highly-divisor-count candidates and measure.

    # Generate candidate exponent patterns for small prime sets
    # We want product(a_i+1) large but coprime fraction reasonable.
    # Try a range of "smooth" exponent vectors.

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Precompute prime powers
    def prime_powers(primes, max_exp):
        pp = {}
        for p in primes:
            vals = [1]
            for e in range(1, max_exp+1):
                vals.append(vals[-1]*p)
            pp[p] = vals
        return pp

    # Search over exponent assignments using a greedy/random approach with budget
    import random
    random.seed(42)

    # Try structured candidates: decreasing exponents
    candidates = []
    # Pattern A: 2^a 3^b 5^c ... decreasing
    for a in range(1, 13):
        for b in range(1, min(a+1, 10)):
            for c in range(1, min(b+1, 8)):
                for d in range(1, min(c+1, 6)):
                    exponents = [a, b, c, d]
                    # extend with 1s
                    n = 1
                    for i, e in enumerate(exponents):
                        n *= small_primes[i] ** e
                    if n <= LIMIT:
                        # try adding more primes with exponent 1
                        for j in range(4, len(small_primes)):
                            if n * small_primes[j] <= LIMIT:
                                n *= small_primes[j]
                                exponents.append(1)
                            else:
                                break
                        candidates.append(exponents[:])

    # Deduplicate
    seen = set()
    unique_candidates = []
    for exps in candidates:
        key = tuple(exps)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(exps)

    # Evaluate candidates, prioritizing by divisor count
    def divisor_count(exps):
        c = 1
        for e in exps:
            c *= (e+1)
        return c

    unique_candidates.sort(key=lambda e: divisor_count(e), reverse=True)

    # Evaluate top candidates within budget
    for exps in unique_candidates:
        if elapsed() > DEADLINE - 5:
            break
        n = 1
        for i, e in enumerate(exps):
            n *= small_primes[i] ** e
        if n > LIMIT:
            continue
        factors = [(small_primes[i], e) for i, e in enumerate(exps)]
        s, nd = tau_perp(n, factors)
        if s > best_score:
            best_n, best_score = n, s

    if elapsed() > DEADLINE - 3:
        return best_n

    # Random search: random exponent vectors
    for _ in range(200000):
        if elapsed() > DEADLINE - 3:
            break
        # random number of primes
        k = random.randint(6, 12)
        exps = []
        for i in range(k):
            # decreasing-ish
            max_e = max(1, 12 - i)
            exps.append(random.randint(1, max_e))
        n = 1
        for i, e in enumerate(exps):
            n *= small_primes[i] ** e
        if n > LIMIT:
            continue
        factors = [(small_primes[i], e) for i, e in enumerate(exps)]
        s, nd = tau_perp(n, factors)
        if s > best_score:
            best_n, best_score = n, s

    return best_n
