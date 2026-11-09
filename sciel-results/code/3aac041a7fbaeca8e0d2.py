"""
Strategy: Dynamic Programming on Prime Sets to Maximize Coprime Adjacent Pairs.

The current program constructs specific n and scores them. This approach:
1. Identifies that for a fixed set of primes P, the problem is to assign exponents
   e_p to maximize the number of coprime adjacent pairs in the sorted divisor list.
2. Observes that for squarefree kernels, the structure of coprime adjacencies is
   determined by the "gaps" in the exponent vectors.
3. Uses a greedy strategy: Start with the smallest primes (2, 3, 5, ...).
   For a given set of primes, the optimal exponents are often heavily skewed
   (e.g., 2^k * 3 * 5 * ...).
4. Instead of random search, we systematically explore the space of prime sets
   and use a local search on the exponents, guided by the fact that increasing
   the exponent of the smallest prime (2) usually yields the highest marginal
   gain in score per unit of log(n) budget, up to a point where adding a new
   prime becomes more efficient.
5. We precompute the score for a candidate n efficiently using a recursive
   generation of divisors that avoids full sorting if possible, or uses a
   faster sorting method. However, the key is to narrow the search space
   significantly by focusing on the structure: n = 2^a * m where m is squarefree
   and composed of the next smallest primes.
"""
# EVOLVE-BLOCK-START
import time
import math
import random

DEADLINE = 145.0
_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def _score_fast(n):
    """
    Calculate the score tau_perp(n) efficiently.
    Factorize n, generate divisors, sort, count coprime pairs.
    """
    if n < 2:
        return 0
    
    # Factorize n
    factors = {}
    temp_n = n
    p = 2
    while p * p <= temp_n:
        if temp_n % p == 0:
            cnt = 0
            while temp_n % p == 0:
                temp_n //= p
                cnt += 1
            factors[p] = cnt
        p += 1 if p == 2 else 2 # Simple odd step, good enough for small primes mostly
    if temp_n > 1:
        factors[temp_n] = 1
        
    # Generate divisors
    divisors = [1]
    for p, e in factors.items():
        new_divs = []
        for d in divisors:
            val = d
            for i in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
        
    divisors.sort()
    
    score = 0
    for i in range(1, len(divisors)):
        if math.gcd(divisors[i-1], divisors[i]) == 1:
            score += 1
    return score

def construct():
    start_time = time.time()
    
    # Initial best: known good structure
    # n = 2^a * 3 * 5 * 7 * 11 * 13 * 17 * 19 ...
    # Let's start with a reasonable guess
    best_n = 2**10 * 3 * 5 * 7 * 11 * 13
    best_score = _score_fast(best_n)
    
    # Strategy:
    # 1. Determine the set of primes to use. Usually the first k primes.
    # 2. For a fixed set of primes, the exponents should be non-increasing?
    #    Actually, for maximizing coprime pairs, we want the divisors to be
    #    "spread out" in the multiplicative lattice such that adjacent integers
    #    in the sorted list are coprime.
    #    
    #    Empirically, the best numbers are of the form 2^a * 3^b * 5^c * ...
    #    with a >= b >= c >= ...
    #    
    #    Let's search over the number of distinct prime factors k.
    #    For each k, we search for the optimal exponents.
    
    max_primes = 15 # 2*3*5*...*47 is around 6e17, so up to 15-16 primes is safe for squarefree part
    
    # Precompute prefix products of primes
    prefix_prod = [1]
    for p in _SMALL_PRIMES:
        prefix_prod.append(prefix_prod[-1] * p)
        
    # We will iterate over possible sets of primes.
    # Most optimal sets are the first k primes.
    # We will also try swapping the largest prime with a slightly larger one to see if it helps.
    
    for k in range(2, min(16, len(_SMALL_PRIMES))):
        if time.time() > DEADLINE:
            break
            
        base_primes = _SMALL_PRIMES[:k]
        
        # The product of the first k primes
        prod_sqfree = prefix_prod[k]
        
        if prod_sqfree > 10**18:
            break
            
        # We have a budget for the exponents.
        # Let n = prod(p_i^{e_i}).
        # log(n) = sum(e_i * log(p_i)) <= log(10^18)
        
        # Heuristic: e_1 (for 2) is the largest, e_2 (for 3) is next, etc.
        # Often e_i = 1 for i > 1, and e_1 is large.
        # Or e_1, e_2 are large, others 1.
        
        # Let's do a local search around the "all 1s" configuration,
        # adding to the exponent of 2 first, then 3, etc.
        
        # Initialize exponents to 1
        exponents = [1] * k
        
        # Current n
        n_curr = prod_sqfree
        s_curr = _score_fast(n_curr)
        
        if s_curr > best_score:
            best_score = s_curr
            best_n = n_curr
            
        # Greedy improvement:
        # While we can increase an exponent without exceeding 10^18:
        # Try increasing exponents in order of "efficiency" or just index order.
        # Since 2 is the smallest, increasing e_1 usually gives the most divisors
        # for the smallest increase in n.
        
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            # Try increasing exponents from smallest prime to largest
            # Why? Because increasing e_1 multiplies n by 2, e_2 by 3, etc.
            # The marginal gain in score per log(n) is likely highest for small primes.
            
            # However, sometimes increasing a larger exponent allows a better structure.
            # Let's try all possible single-step increases and pick the best one.
            
            best_gain = -1
            best_idx = -1
            best_new_n = n_curr
            best_new_s = s_curr
            best_new_exp = exponents[:]
            
            for i in range(k):
                p = base_primes[i]
                new_n = n_curr * p
                if new_n > 10**18:
                    continue
                new_exp = exponents[:]
                new_exp[i] += 1
                s_new = _score_fast(new_n)
                
                # We want to maximize score.
                # If s_new > s_curr, it's a gain.
                # We might also consider if a larger jump is better, but single steps are safe.
                
                if s_new > best_new_s:
                    best_new_s = s_new
                    best_new_n = new_n
                    best_new_exp = new_exp
                    best_idx = i
                    best_gain = s_new - s_curr
            
            if best_gain > 0:
                n_curr = best_new_n
                s_curr = best_new_s
                exponents = best_new_exp
                changed = True
                
                if s_curr > best_score:
                    best_score = s_curr
                    best_n = n_curr
            else:
                # No single step improves score?
                # Maybe we need to decrease one exponent to increase another?
                # Or maybe we should add a new prime?
                # For now, stop this local search.
                break
                
        # Try adding the next prime?
        # If we have space, adding a new prime p_{k+1} to the squarefree part
        # usually increases the score.
        # This is handled by the outer loop over k.
        
    # Additional refinement: Random perturbations of the best found structure
    # to escape local optima in the exponent space.
    
    if time.time() < DEADLINE - 10:
        # Extract best structure
        # Factor best_n to get primes and exponents
        factors = {}
        temp_n = best_n
        p = 2
        while p * p <= temp_n:
            if temp_n % p == 0:
                cnt = 0
                while temp_n % p == 0:
                    temp_n //= p
                    cnt += 1
                factors[p] = cnt
            p += 1 if p == 2 else 2
        if temp_n > 1:
            factors[temp_n] = 1
            
        primes = list(factors.keys())
        exps = list(factors.values())
        
        # Shuffle primes? No, the set of primes matters.
        # Let's try randomizing the exponents slightly around the best found.
        
        for _ in range(1000):
            if time.time() > DEADLINE:
                break
                
            new_exps = exps[:]
            # Perturb
            for i in range(len(new_exps)):
                if random.random() < 0.5:
                    new_exps[i] = max(1, new_exps[i] + random.randint(-2, 2))
            
            n_new = 1
            for p, e in zip(primes, new_exps):
                n_new *= (p ** e)
                
            if n_new <= 10**18 and n_new >= 2:
                s_new = _score_fast(n_new)
                if s_new > best_score:
                    best_score = s_new
                    best_n = n_new
                    # Update reference
                    factors = {}
                    temp_n = best_n
                    p = 2
                    while p * p <= temp_n:
                        if temp_n % p == 0:
                            cnt = 0
                            while temp_n % p == 0:
                                temp_n //= p
                                cnt += 1
                            factors[p] = cnt
                        p += 1 if p == 2 else 2
                    if temp_n > 1:
                        factors[temp_n] = 1
                    primes = list(factors.keys())
                    exps = list(factors.values())

    return best_n
# EVOLVE-BLOCK-END
