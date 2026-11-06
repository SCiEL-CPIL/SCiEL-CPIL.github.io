import math
import time
from math import gcd
from itertools import product, combinations

def _get_primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

_PRIMES = _get_primes(5000)

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

def _count_coprime_adjacent(divs):
    """Count coprime adjacent pairs in sorted divisor list."""
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _search():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    LIMIT = 10**18
    
    primes = _PRIMES
    
    # First, find the maximum k such that product of first k primes <= LIMIT
    max_k = 0
    prod = 1
    for i, p in enumerate(primes):
        if prod * p > LIMIT:
            break
        prod *= p
        max_k = i + 1
    
    # Start with primorial of max_k primes
    best_n = 1
    best_score = 0
    best_factorization = []
    
    # Evaluate primorial
    prod = 1
    factorization = []
    for i in range(max_k):
        prod *= primes[i]
        factorization.append((primes[i], 1))
    best_n = prod
    best_score = _tau_perp_fast(prod, factorization)
    best_factorization = factorization[:]
    
    # Local search: try swapping one prime with a larger prime
    # This can sometimes increase the coprime count by changing the divisor structure
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        improved = False
        # Try swapping each prime position with a larger prime
        for pos in range(max_k):
            if time.time() > deadline - 5:
                break
            
            current_prime = best_factorization[pos][0]
            # Try replacing with primes from a window
            # We want primes larger than current but not too large
            start_idx = primes.index(current_prime) + 1
            end_idx = min(start_idx + 50, len(primes))
            
            for new_idx in range(start_idx, end_idx):
                new_prime = primes[new_idx]
                
                # Check if product stays within limit
                # Remove old prime, add new prime
                new_prod = 1
                new_factorization = []
                valid = True
                for j, (p, e) in enumerate(best_factorization):
                    if j == pos:
                        if new_prod * new_prime > LIMIT:
                            valid = False
                            break
                        new_prod *= new_prime
                        new_factorization.append((new_prime, 1))
                    else:
                        if new_prod * p > LIMIT:
                            valid = False
                            break
                        new_prod *= p
                        new_factorization.append((p, 1))
                
                if not valid:
                    continue
                
                # Check if new_prime is not already in the set
                if any(p == new_prime for p, e in best_factorization if (p, e) != (current_prime, 1)):
                    continue
                
                # Evaluate
                score = _tau_perp_fast(new_prod, new_factorization)
                if score > best_score:
                    best_score = score
                    best_n = new_prod
                    best_factorization = new_factorization[:]
                    improved = True
                    break  # Restart from this improved position
        
        if not improved:
            break
    
    # Also try: for smaller k, explore more prime combinations
    # Sometimes fewer primes with better structure beats more primes
    for k in range(max(2, max_k - 3), max_k):
        if time.time() > deadline - 3:
            break
        
        # Try primorial of first k primes as baseline
        prod = 1
        factorization = []
        for i in range(k):
            prod *= primes[i]
            factorization.append((primes[i], 1))
        
        score = _tau_perp_fast(prod, factorization)
        if score > best_score:
            best_score = score
            best_n = prod
            best_factorization = factorization[:]
    
    # Try some non-squarefree candidates: p^2 * q * r * ...
    # Squaring a small prime adds divisors but may increase coprime pairs
    for exp_pos in range(min(4, max_k)):
        if time.time() > deadline - 3:
            break
        for k in range(max_k - 1, max_k + 1):
            if k > len(primes):
                break
            prod = 1
            factorization = []
            valid = True
            for i in range(k):
                exp = 2 if i == exp_pos else 1
                if prod * (primes[i] ** exp) > LIMIT:
                    valid = False
                    break
                prod *= primes[i] ** exp
                factorization.append((primes[i], exp))
            if not valid:
                continue
            
            num_divs = 1
            for _, e in factorization:
                num_divs *= (e + 1)
            if num_divs > 500000:
                continue
            
            score = _tau_perp_fast(prod, factorization)
            if score > best_score:
                best_score = score
                best_n = prod
    
    return best_n

def construct():
    return _search()
