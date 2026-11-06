import math
from itertools import combinations
from functools import lru_cache

def _get_divisors(num):
    """Generate all divisors of num in unsorted order."""
    divisors = [1]
    i = 2
    while i * i <= num:
        if num % i == 0:
            power = 1
            while num % i == 0:
                num //= i
                power *= i
                for d in list(divisors):
                    divisors.append(d * power)
        i += 1
    if num > 1:
        for d in list(divisors):
            divisors.append(d * num)
    return divisors

def _score_n(n):
    """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
    divs = _get_divisors(n)
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _generate_primes(limit):
    """Generate all primes up to limit using sieve."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    import time
    start_time = time.time()
    time_limit = 110
    
    # We focus on squarefree numbers with 14 or 15 distinct prime factors.
    # The product of the first 15 primes is ~6.15e17, which fits.
    # The product of the first 16 primes is ~3.2e19, which is too large.
    # So max distinct primes is 15.
    
    primes = _generate_primes(100)
    
    best_n = 2
    best_score = _score_n(2)
    
    # Evaluate primorials first to get a strong baseline
    prod = 1
    for p in primes:
        prod *= p
        if prod > 10**18:
            break
        s = _score_n(prod)
        if s > best_score:
            best_score = s
            best_n = prod

    # Now, search for better subsets of size 15 (and maybe 14)
    # We consider the first 20 primes.
    # Combinations of 15 from 20 is 15504.
    # Combinations of 14 from 20 is 38760.
    # Total ~54k evaluations.
    # Each eval is fast (<1ms). Total time ~1-2 minutes?
    # 54000 * 1ms = 54s. This is tight but should fit in 110s.
    # To be safe, we'll prioritize size 15.
    
    # Optimization: Pre-filter combinations by product.
    # We can prune if the product of the smallest 15 in the subset exceeds 10^18?
    # No, we just check the product.
    
    # We will iterate over combinations of the first 20 primes.
    candidate_primes = primes[:20]
    
    # Try size 15
    for combo in combinations(candidate_primes, 15):
        if time.time() - start_time > time_limit:
            break
        n_val = 1
        for p in combo:
            n_val *= p
        if n_val > 10**18:
            continue
        s = _score_n(n_val)
        if s > best_score:
            best_score = s
            best_n = n_val

    # Try size 14 (in case 14 primes with smaller values beat 15 primes with larger values? 
    # Unlikely, but 38k evals is manageable if we have time)
    if time.time() - start_time < time_limit - 30:
        for combo in combinations(candidate_primes, 14):
            if time.time() - start_time > time_limit - 10:
                break
            n_val = 1
            for p in combo:
                n_val *= p
            if n_val > 10**18:
                continue
            s = _score_n(n_val)
            if s > best_score:
                best_score = s
                best_n = n_val

    return best_n
