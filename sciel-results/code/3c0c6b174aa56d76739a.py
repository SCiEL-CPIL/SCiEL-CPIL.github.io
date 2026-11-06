"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START

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
    return [i for i, v in enumerate(is_prime) if v]

def _tau_perp(n):
    """Count adjacent coprime divisor pairs."""
    if n == 1:
        return 0
    # Get all divisors
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.sort()
    if len(divs) < 2:
        return 0
    count = 0
    for j in range(len(divs) - 1):
        if math.gcd(divs[j], divs[j+1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    LIMIT = 10**18
    
    # Strategy: try various combinations of primes
    # Key insight: for maximizing coprime adjacent pairs, we want
    # divisors that are "spread out" so adjacent ones share few factors
    
    # Precompute primes up to a reasonable limit
    primes = _get_primes(10000)
    
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19  # seed
    best_score = _tau_perp(best_n)
    
    # Try different sets of primes
    # For square-free numbers, try various prime combinations
    # The product must be <= 10^18
    
    # First, try primorials of different lengths
    prod = 1
    for i, p in enumerate(primes):
        if prod * p > LIMIT:
            break
        prod *= p
        n_candidate = prod
        score = _tau_perp(n_candidate)
        if score > best_score:
            best_score = score
            best_n = n_candidate
    
    # Try replacing some primes with larger ones to create gaps
    # This can create more coprime adjacent pairs
    base_primes = primes[:10]  # 2,3,5,7,11,13,17,19,23,29
    
    # Try various combinations: use small primes plus one or two large primes
    # The large primes create "jumps" in the divisor sequence
    
    # For each subset of small primes, try adding large primes
    small_set = primes[:8]  # up to 19
    base_prod = 1
    for p in small_set:
        base_prod *= p
    
    # Try replacing the largest small prime with a larger one
    for replace_idx in range(3, 8):  # try replacing 7,11,13,17,19
        for new_p in primes[replace_idx+1:min(len(primes), replace_idx+200)]:
            cand_prod = 1
            for j, p in enumerate(small_set):
                if j == replace_idx:
                    cand_prod *= new_p
                else:
                    cand_prod *= p
            if cand_prod > LIMIT:
                break
            n_candidate = cand_prod
            if time.time() > DEADLINE:
                break
            score = _tau_perp(n_candidate)
            if score > best_score:
                best_score = score
                best_n = n_candidate
    
    # Try with fewer small primes but more spread out
    # e.g., 2 * 3 * 5 * large_prime * another_large_prime
    for i1 in range(1, 4):
        for i2 in range(i1+1, 6):
            for i3 in range(i2+1, 8):
                base = primes[i1] * primes[i2] * primes[i3]
                if base > LIMIT:
                    continue
                # Try adding 2 and a large prime
                for large_p in primes[8:50]:
                    cand = 2 * base * large_p
                    if cand > LIMIT:
                        break
                    if time.time() > DEADLINE:
                        break
                    score = _tau_perp(cand)
                    if score > best_score:
                        best_score = score
                        best_n = cand
    
    # Try numbers with prime powers mixed in
    # e.g., 2^a * 3^b * 5^c * ... with some exponents > 1
    # This increases divisor count but may reduce coprime adjacencies
    
    # Try: 2^1 * 3^1 * 5^1 * 7^1 * 11^1 * 13^1 * p_large
    # where p_large is chosen to create good gaps
    base6 = 2*3*5*7*11*13
    for p in primes[7:200]:  # starting from 19
        cand = base6 * p
        if cand > LIMIT:
            break
        if time.time() > DEADLINE:
            break
        score = _tau_perp(cand)
        if score > best_score:
            best_score = score
            best_n = cand
    
    # Try: 2*3*5*7*11*p1*p2 where p1, p2 are larger primes
    base5 = 2*3*5*7*11
    for i in range(6, min(30, len(primes))):
        p1 = primes[i]
        for j in range(i+1, min(50, len(primes))):
            p2 = primes[j]
            cand = base5 * p1 * p2
            if cand > LIMIT:
                break
            if time.time() > DEADLINE:
                break
            score = _tau_perp(cand)
            if score > best_score:
                best_score = score
                best_n = cand
    
    # Check deadline
    if time.time() > DEADLINE:
        return best_n
    
    # Try more exotic combinations
    # 2 * 3 * 5 * p1 * p2 * p3 with p1,p2,p3 being larger
    base3 = 2*3*5
    for i in range(3, min(15, len(primes))):
        p1 = primes[i]
        for j in range(i+1, min(25, len(primes))):
            p2 = primes[j]
            for k in range(j+1, min(40, len(primes))):
                p3 = primes[k]
                cand = base3 * p1 * p2 * p3
                if cand > LIMIT:
                    break
                if time.time() > DEADLINE:
                    break
                score = _tau_perp(cand)
                if score > best_score:
                    best_score = score
                    best_n = cand
    
    return best_n

# EVOLVE-BLOCK-END
