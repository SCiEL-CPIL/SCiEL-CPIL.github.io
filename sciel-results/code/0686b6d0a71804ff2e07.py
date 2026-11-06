import time
import math
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START

def _divisors(n):
    """Generate all divisors of n in sorted order."""
    # Factorize
    factors = {}
    tmp = n
    d = 2
    while d * d <= tmp:
        while tmp % d == 0:
            factors[d] = factors.get(d, 0) + 1
            tmp //= d
        d += 1
    if tmp > 1:
        factors[tmp] = factors.get(tmp, 0) + 1
    
    # Generate divisors
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        for d in divs:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divs = new_divs
    divs.sort()
    return divs

def _score(n):
    """Compute tau_perp(n): number of adjacent divisor pairs with gcd 1."""
    divs = _divisors(n)
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _factorize(n):
    """Return list of (prime, exponent) pairs."""
    factors = []
    tmp = n
    d = 2
    while d * d <= tmp:
        if tmp % d == 0:
            e = 0
            while tmp % d == 0:
                tmp //= d
                e += 1
            factors.append((d, e))
        d += 1
    if tmp > 1:
        factors.append((tmp, 1))
    return factors

def _value_from_factors(factors):
    """Compute n from list of (prime, exponent)."""
    n = 1
    for p, e in factors:
        n *= p ** e
    return n

def _try_local_search(start_factors, deadline, best_n, best_score):
    """Local search over exponent changes and prime swaps."""
    factors = list(start_factors)
    current_n = _value_from_factors(factors)
    current_score = _score(current_n)
    
    if current_score > best_score:
        best_score = current_score
        best_n = current_n
    
    # Primes to consider (first 20 primes)
    primes_pool = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(factors)):
            if time.time() > deadline - 5:
                break
            p, e = factors[i]
            # Try increasing exponent
            new_factors = list(factors)
            new_factors[i] = (p, e + 1)
            new_n = _value_from_factors(new_factors)
            if new_n <= 10**18:
                s = _score(new_n)
                if s > current_score:
                    factors = new_factors
                    current_n = new_n
                    current_score = s
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_n = current_n
            # Try decreasing exponent
            if e > 1:
                new_factors = list(factors)
                new_factors[i] = (p, e - 1)
                new_n = _value_from_factors(new_factors)
                s = _score(new_n)
                if s > current_score:
                    factors = new_factors
                    current_n = new_n
                    current_score = s
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_n = current_n
            # Try swapping prime with another
            for j, (p2, e2) in enumerate(factors):
                if j == i:
                    continue
                # Swap primes i and j (keep exponents)
                new_factors = list(factors)
                new_factors[i] = (p2, e)
                new_factors[j] = (p, e2)
                new_n = _value_from_factors(new_factors)
                if new_n <= 10**18:
                    s = _score(new_n)
                    if s > current_score:
                        factors = new_factors
                        current_n = new_n
                        current_score = s
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
        # Try adding a new prime factor
        if time.time() < deadline - 5:
            for p in primes_pool:
                if any(p == pf for pf, _ in factors):
                    continue
                new_factors = factors + [(p, 1)]
                new_n = _value_from_factors(new_factors)
                if new_n <= 10**18:
                    s = _score(new_n)
                    if s > current_score:
                        factors = new_factors
                        current_n = new_n
                        current_score = s
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
        # Try removing a prime factor
        if time.time() < deadline - 5 and len(factors) > 2:
            for i in range(len(factors)):
                new_factors = factors[:i] + factors[i+1:]
                new_n = _value_from_factors(new_factors)
                s = _score(new_n)
                if s > current_score:
                    factors = new_factors
                    current_n = new_n
                    current_score = s
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_n = current_n
    
    return best_n, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with 15th primorial
    primes_15 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    start_factors = [(p, 1) for p in primes_15]
    start_n = 1
    for p in primes_15:
        start_n *= p
    
    best_n = start_n
    best_score = _score(start_n)
    
    # Local search from primorial
    best_n, best_score = _try_local_search(start_factors, deadline, best_n, best_score)
    
    # Try other starting points
    # Start with fewer primes but higher exponents
    if time.time() < deadline - 10:
        for num_primes in [10, 11, 12, 13, 14]:
            if time.time() > deadline - 10:
                break
            pf = [(p, 1) for p in primes_15[:num_primes]]
            best_n, best_score = _try_local_search(pf, deadline, best_n, best_score)
    
    # Try structures with exponents on small primes
    if time.time() < deadline - 10:
        # 2^a * 3^b * 5^c * ... with some exponents > 1
        test_factors = [(2, 3), (3, 2), (5, 2), (7, 2), (11, 1), (13, 1), (17, 1), (19, 1), (23, 1), (29, 1), (31, 1)]
        test_n = _value_from_factors(test_factors)
        if test_n <= 10**18:
            best_n, best_score = _try_local_search(test_factors, deadline, best_n, best_score)
    
    if time.time() < deadline - 5:
        test_factors2 = [(2, 4), (3, 3), (5, 2), (7, 2), (11, 1), (13, 1), (17, 1), (19, 1), (23, 1), (29, 1)]
        test_n2 = _value_from_factors(test_factors2)
        if test_n2 <= 10**18:
            best_n, best_score = _try_local_search(test_factors2, deadline, best_n, best_score)
    
    return best_n

# EVOLVE-BLOCK-END
