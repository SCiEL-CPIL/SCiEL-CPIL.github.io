"""Maximize coprime adjacent divisor pairs for n <= 10^18."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _score(n):
    """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
    if n < 2:
        return 0
    # Factorize n
    factors = []
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            cnt = 0
            while temp % p == 0:
                cnt += 1
                temp //= p
            factors.append((p, cnt))
        p += 1 if p == 2 else 2
    if temp > 1:
        factors.append((temp, 1))
    
    # Generate all divisors
    divisors = [1]
    for p, e in factors:
        new_divs = []
        for d in divisors:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divisors = new_divs
    
    divisors.sort()
    
    # Count adjacent coprime pairs
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            count += 1
    return count

def _try_primes(prime_list):
    """Try a set of primes (squarefree product) and return (n, score)."""
    n = 1
    for p in prime_list:
        n *= p
    if n > 10**18:
        return None, -1
    s = _score(n)
    return n, s

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # First, try the primorial of first 15 primes (known good: score ~21.9)
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    best_n, best_score = _try_primes(base_primes)
    
    # Try variations: swap out some primes for larger ones to create bigger gaps
    # Generate candidate primes up to a reasonable bound
    def is_prime(x):
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
    
    # Generate primes up to ~1000 for swapping
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    candidate_primes = list(all_primes)
    p = 53
    while p <= 2000:
        if is_prime(p):
            candidate_primes.append(p)
        p += 2
    
    # Try swapping individual primes for larger ones
    for i in range(len(base_primes)):
        if time.time() > deadline:
            break
        for j in range(len(candidate_primes)):
            if candidate_primes[j] == base_primes[i]:
                continue
            new_primes = base_primes[:i] + [candidate_primes[j]] + base_primes[i+1:]
            n, s = _try_primes(new_primes)
            if s > best_score:
                best_score = s
                best_n = n
    
    # Try swapping pairs of primes
    for i in range(len(base_primes)):
        if time.time() > deadline:
            break
        for j in range(i+1, len(base_primes)):
            if time.time() > deadline:
                break
            for a in range(len(candidate_primes)):
                if time.time() > deadline:
                    break
                for b in range(a+1, len(candidate_primes)):
                    new_primes = base_primes[:i] + [candidate_primes[a], candidate_primes[b]] + base_primes[j+1:]
                    # Remove the two original primes and insert the new ones, keeping sorted
                    new_primes = sorted(set(new_primes))
                    if len(new_primes) != len(base_primes):
                        continue
                    n, s = _try_primes(new_primes)
                    if s > best_score:
                        best_score = s
                        best_n = n
    
    # Try with 14 primes (fewer primes, can use larger ones)
    if time.time() < deadline - 10:
        for i in range(len(base_primes)):
            if time.time() > deadline:
                break
            reduced = base_primes[:i] + base_primes[i+1:]
            for a in range(len(candidate_primes)):
                if time.time() > deadline:
                    break
                new_primes = sorted(reduced + [candidate_primes[a]])
                if len(new_primes) != 14:
                    continue
                n, s = _try_primes(new_primes)
                if s > best_score:
                    best_score = s
                    best_n = n
    
    # Try 16 primes if possible (product of first 16 primes > 10^18, but maybe with smaller selection)
    # Actually first 16 primes product is 3.29e19 > 10^18, so can't use 16 smallest
    # But maybe 16 primes with some very small? No, smallest 16 already exceed.
    
    return best_n if best_n is not None else 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19

# EVOLVE-BLOCK-END
