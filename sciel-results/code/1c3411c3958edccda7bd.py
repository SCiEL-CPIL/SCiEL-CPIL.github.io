import math
import time
from math import gcd
from itertools import product
import random

def _get_primes(limit):
    """Sieve of Eratosthenes."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

_PRIMES = _get_primes(1000)

def _tau_perp_fast(n, factorization):
    """Compute tau_perp using factorization for speed.
    factorization: list of (prime, exponent)
    """
    # Generate all divisors from factorization
    divs = [1]
    for p, e in factorization:
        new_divs = []
        for d in divs:
            val = d
            for exp in range(e + 1):
                new_divs.append(val)
                val *= p
        divs = new_divs
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _search():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    LIMIT = 10**18
    
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    best_score = _tau_perp_fast(best_n, [(2,1),(3,1),(5,1),(7,1),(11,1),(13,1),(17,1),(19,1)])
    
    primes = _PRIMES
    
    # Strategy 1: Try squarefree numbers with first k primes (primorials)
    for k in range(2, 20):
        if time.time() > deadline - 5:
            break
        prod = 1
        factorization = []
        valid = True
        for i in range(k):
            if prod * primes[i] > LIMIT:
                valid = False
                break
            prod *= primes[i]
            factorization.append((primes[i], 1))
        if not valid or k < 2:
            continue
        score = _tau_perp_fast(prod, factorization)
        if score > best_score:
            best_score = score
            best_n = prod
    
    # Strategy 2: Explore numbers with various exponent patterns
    for k in range(2, 15):
        if time.time() > deadline - 3:
            break
        
        patterns = []
        
        # All ones (squarefree)
        patterns.append([1]*k)
        
        # Various patterns with one prime having higher exponent
        for e1 in [2, 3, 4, 5]:
            for pos in range(min(k, 4)):
                pat = [1]*k
                pat[pos] = e1
                patterns.append(pat)
        
        # Two primes with higher exponents
        for e1 in [2, 3]:
            for e2 in [2, 3]:
                for p1 in range(min(k, 3)):
                    for p2 in range(p1+1, min(k, 4)):
                        pat = [1]*k
                        pat[p1] = e1
                        pat[p2] = e2
                        patterns.append(pat)
        
        # Try each pattern
        seen = set()
        for pattern in patterns:
            if time.time() > deadline - 2:
                break
            key = tuple(pattern)
            if key in seen:
                continue
            seen.add(key)
            
            prod = 1
            factorization = []
            valid = True
            for i in range(k):
                p = primes[i]
                e = pattern[i]
                pe = p ** e
                if prod * pe > LIMIT:
                    valid = False
                    break
                prod *= pe
                factorization.append((p, e))
            if not valid:
                continue
            
            num_divs = 1
            for _, e in factorization:
                num_divs *= (e + 1)
            if num_divs > 200000:
                continue
            
            score = _tau_perp_fast(prod, factorization)
            if score > best_score:
                best_score = score
                best_n = prod
    
    # Strategy 3: Random search with more diverse structures
    if time.time() < deadline - 10:
        random.seed(42)
        
        for trial in range(5000):
            if time.time() > deadline - 2:
                break
            
            # Random number of primes
            k = random.randint(3, 14)
            
            # Choose primes: mostly small
            chosen_primes = []
            for i in range(k):
                max_idx = min(i + random.randint(1, 3), 40)
                idx = random.randint(0, max_idx)
                chosen_primes.append(primes[idx])
            
            chosen_primes = sorted(set(chosen_primes))
            if len(chosen_primes) < 2:
                continue
            k = len(chosen_primes)
            
            # Random exponents, biased toward 1
            factorization = []
            prod = 1
            valid = True
            for p in chosen_primes:
                r = random.random()
                if r < 0.75:
                    e = 1
                elif r < 0.95:
                    e = 2
                else:
                    e = random.randint(3, 5)
                pe = p ** e
                if prod * pe > LIMIT:
                    valid = False
                    break
                prod *= pe
                factorization.append((p, e))
            
            if not valid or len(factorization) < 2:
                continue
            
            num_divs = 1
            for _, e in factorization:
                num_divs *= (e + 1)
            if num_divs > 200000:
                continue
            
            score = _tau_perp_fast(prod, factorization)
            if score > best_score:
                best_score = score
                best_n = prod
    
    return best_n

def construct():
    return _search()
# EVOLVE-BLOCK-END
