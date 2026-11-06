"""
Constructs an integer n <= 10^18 maximizing tau_perp(n).

Approach: Model the sorted divisor lattice as a sequence of intervals where
adjacent coprime pairs correspond to transitions between disjoint prime supports.
Use a greedy interval-merging heuristic to select prime exponents that maximize
the density of support-disjoint boundaries within the 10^18 budget.
"""

import math
from itertools import combinations

def _divisors(n):
    """Generate all divisors of n in sorted order."""
    divs = [1]
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            new_divs = []
            for d in divs:
                mult = 1
                for e in range(exp + 1):
                    new_divs.append(d * mult)
                    mult *= p
            divs = new_divs
        p += 1 if p == 2 else 2
    if temp > 1:
        new_divs = []
        for d in divs:
            new_divs.append(d)
            new_divs.append(d * temp)
        divs = new_divs
    divs.sort()
    return divs

def _tau_perp(n):
    """Count adjacent coprime pairs in sorted divisor list of n."""
    if n < 2:
        return 0
    divs = _divisors(n)
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def _is_prime(x):
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0 or x % 3 == 0:
        return False
    i = 5
    while i * i <= x:
        if x % i == 0 or x % (i + 2) == 0:
            return False
        i += 6
    return True

def _primes_up_to(limit):
    """Generate primes up to limit using simple sieve."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

def _factorize(n):
    """Return list of (prime, exponent) pairs for n."""
    factors = []
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            factors.append((p, exp))
        p += 1 if p == 2 else 2
    if temp > 1:
        factors.append((temp, 1))
    return factors

def _divisor_count_from_factors(factors):
    """Compute number of divisors from factorization."""
    count = 1
    for _, exp in factors:
        count *= (exp + 1)
    return count

def _tau_perp_from_factors(factors):
    """Compute tau_perp from factorization without generating all divisors if possible."""
    # For small divisor counts, generate and count
    n = 1
    for p, e in factors:
        n *= p ** e
    return _tau_perp(n)

def construct():
    """
    Greedy interval-merging heuristic:
    1. Start with squarefree products of smallest primes
    2. Greedily add prime powers where they increase tau_perp density
    3. Explore local neighborhoods of promising candidates
    """
    all_primes = _primes_up_to(500)
    
    best_n = 2
    best_score = 1  # n=2 has divisors [1,2], one adjacent pair, gcd=1
    
    # Phase 1: Explore squarefree products of first k primes
    prod = 1
    for k in range(1, 20):
        if k >= len(all_primes):
            break
        prod *= all_primes[k - 1]
        if prod > 10**18:
            break
        score = _tau_perp(prod)
        if score > best_score:
            best_score = score
            best_n = prod
    
    # Phase 2: Greedy addition of prime powers
    # For the best squarefree base, try adding powers of small primes
    base_factors = _factorize(best_n)
    base_n_val = best_n
    base_score_val = best_score
    
    # Try adding powers to each prime in the factorization
    for trial_p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        for exp in range(2, 30):
            candidate = base_n_val * (trial_p ** (exp - 1))  # multiply by p^(exp-1) since p^1 already there
            if candidate > 10**18:
                break
            if candidate == base_n_val:
                continue
            score = _tau_perp(candidate)
            if score > best_score:
                best_score = score
                best_n = candidate
    
    # Phase 3: Local search - try replacing one prime with another
    factors = _factorize(best_n)
    for i, (p, e) in enumerate(factors):
        # Try replacing p^e with q^f for various q, f
        for q in all_primes:
            if q == p:
                continue
            for f in range(1, 10):
                candidate = best_n // (p ** e) * (q ** f)
                if candidate > 10**18 or candidate < 2:
                    continue
                if candidate == best_n:
                    continue
                score = _tau_perp(candidate)
                if score > best_score:
                    best_score = score
                    best_n = candidate
    
    # Phase 4: Try constructions with 2^a * product of other small primes
    # This often gives good density because 2 creates many boundaries
    for a in range(1, 60):
        # Start with 2^a, then greedily add smallest primes that fit
        candidate = 2 ** a
        for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73]:
            if candidate * p > 10**18:
                break
            candidate *= p
        if candidate >= 2 and candidate <= 10**18:
            score = _tau_perp(candidate)
            if score > best_score:
                best_score = score
                best_n = candidate
    
    # Phase 5: More systematic search over (2^a, 3^b, rest squarefree)
    for a in range(1, 40):
        for b in range(0, 15):
            base = (2 ** a) * (3 ** b)
            if base > 10**18:
                break
            # Add smallest primes from 5 onward
            candidate = base
            for p in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
                if candidate * p > 10**18:
                    break
                candidate *= p
            if candidate >= 2 and candidate <= 10**18:
                score = _tau_perp(candidate)
                if score > best_score:
                    best_score = score
                    best_n = candidate
    
    return best_n
