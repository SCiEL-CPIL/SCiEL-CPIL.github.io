"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import reduce

def _sieve(n):
    """Generate primes up to n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

def _divisors_from_factors(primes, exponents):
    """Generate all divisors of a number given its prime factorization."""
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            for i in range(e + 1):
                new_divs.append(d * (p ** i))
        divisors = new_divs
    return sorted(divisors)

def _compute_score(divisors):
    """Count adjacent coprime pairs in sorted divisor list."""
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count

def _try_candidate(primes, exponents):
    """Try a candidate number, return (score, n) or (0, 0) if invalid."""
    n = 1
    for p, e in zip(primes, exponents):
        n *= p ** e
    if n < 2 or n > 10**18:
        return (0, 0)
    divs = _divisors_from_factors(primes, exponents)
    score = _compute_score(divs)
    return (score, n)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    # First, try the known good candidate from cell findings
    # 2^5 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    primes_base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    exponents_base = [5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    best_score, best_n = _try_candidate(primes_base, exponents_base)
    
    if time.time() > deadline:
        return best_n if best_n >= 2 else 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    
    # Try variations: different exponent combinations for small primes
    # The idea: give more exponents to smaller primes to increase divisor count
    # while staying within 10^18
    
    # Generate primes up to ~200
    all_primes = _sieve(200)
    
    # Strategy: try different sets of primes and exponent patterns
    # For each set of k primes, try various exponent distributions
    
    # First, determine how many primes we can use
    # Product of first k primes must be <= 10^18
    prod = 1
    max_primes = 0
    for i, p in enumerate(all_primes):
        if prod * p <= 10**18:
            prod *= p
            max_primes = i + 1
        else:
            break
    
    # Try different numbers of primes and exponent patterns
    # Focus on using many primes with small exponents
    
    # Pattern 1: All exponents 1 (squarefree) with max primes
    if max_primes >= 15:
        score, n = _try_candidate(all_primes[:15], [1] * 15)
        if score > best_score:
            best_score, best_n = score, n
    
    # Pattern 2: 2^a * 3^b * rest^1
    # Try different (a, b) combinations
    for a in range(1, 8):
        for b in range(1, 5):
            # Determine how many additional primes we can fit
            base_val = (2 ** a) * (3 ** b)
            remaining = 10**18 // base_val
            # Count how many primes from 5 onwards we can multiply
            count = 0
            prod_rem = 1
            for p in all_primes[2:]:  # skip 2 and 3
                if prod_rem * p <= remaining:
                    prod_rem *= p
                    count += 1
                else:
                    break
            if count < 10:
                continue
            primes_try = [2, 3] + all_primes[2:2 + count]
            exponents_try = [a, b] + [1] * count
            score, n = _try_candidate(primes_try, exponents_try)
            if score > best_score:
                best_score, best_n = score, n
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Pattern 3: 2^a * 3^b * 5^c * rest^1
    if time.time() < deadline:
        for a in range(1, 7):
            for b in range(1, 4):
                for c in range(1, 3):
                    base_val = (2 ** a) * (3 ** b) * (5 ** c)
                    remaining = 10**18 // base_val
                    count = 0
                    prod_rem = 1
                    for p in all_primes[3:]:  # skip 2, 3, 5
                        if prod_rem * p <= remaining:
                            prod_rem *= p
                            count += 1
                        else:
                            break
                    if count < 10:
                        continue
                    primes_try = [2, 3, 5] + all_primes[3:3 + count]
                    exponents_try = [a, b, c] + [1] * count
                    score, n = _try_candidate(primes_try, exponents_try)
                    if score > best_score:
                        best_score, best_n = score, n
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    # Pattern 4: Try with 4th prime also having exponent > 1
    if time.time() < deadline:
        for a in range(1, 6):
            for b in range(1, 4):
                for c in range(1, 3):
                    for d in range(1, 2):
                        base_val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                        remaining = 10**18 // base_val
                        count = 0
                        prod_rem = 1
                        for p in all_primes[4:]:  # skip 2, 3, 5, 7
                            if prod_rem * p <= remaining:
                                prod_rem *= p
                                count += 1
                            else:
                                break
                        if count < 10:
                            continue
                        primes_try = [2, 3, 5, 7] + all_primes[4:4 + count]
                        exponents_try = [a, b, c, d] + [1] * count
                        score, n = _try_candidate(primes_try, exponents_try)
                        if score > best_score:
                            best_score, best_n = score, n
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    # Final check
    if best_n < 2:
        best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    
    return best_n
# EVOLVE-BLOCK-END
