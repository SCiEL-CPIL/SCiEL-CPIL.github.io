"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def tau_perp_fast(primes):
    """Compute tau_perp for squarefree n given its sorted prime list."""
    k = len(primes)
    divs = [1]
    for p in primes:
        divs = divs + [d * p for d in divs]
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    limit = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Find max number of primes
    max_k = 0
    prod = 1
    for p in primes:
        if prod * p > limit:
            break
        prod *= p
        max_k += 1
    
    base_primes = primes[:max_k]
    best_n = 1
    for p in base_primes:
        best_n *= p
    best_score = tau_perp_fast(base_primes)
    
    # Single swaps
    for i in range(max_k):
        if time.time() > deadline:
            break
        prod_without = best_n // base_primes[i]
        for new_p in primes[max_k:]:
            if prod_without * new_p > limit:
                break
            new_primes = base_primes[:i] + [new_p] + base_primes[i+1:]
            new_primes.sort()
            score = tau_perp_fast(new_primes)
            if score > best_score:
                best_score = score
                best_n = prod_without * new_p
    
    # Double swaps
    for i in range(max_k):
        if time.time() > deadline:
            break
        for j in range(i + 1, max_k):
            if time.time() > deadline:
                break
            prod_without = best_n // base_primes[i] // base_primes[j]
            for new_p1 in primes[max_k:]:
                if time.time() > deadline:
                    break
                if prod_without * new_p1 > limit:
                    break
                for new_p2 in primes[primes.index(new_p1)+1:]:
                    if time.time() > deadline:
                        break
                    if prod_without * new_p1 * new_p2 > limit:
                        break
                    new_primes = [p for idx, p in enumerate(base_primes) if idx != i and idx != j]
                    new_primes.append(new_p1)
                    new_primes.append(new_p2)
                    new_primes.sort()
                    n_candidate = prod_without * new_p1 * new_p2
                    score = tau_perp_fast(new_primes)
                    if score > best_score:
                        best_score = score
                        best_n = n_candidate
    
    return best_n
# EVOLVE-BLOCK-END
