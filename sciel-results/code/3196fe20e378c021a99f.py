import math
import time
from functools import lru_cache

def _score_from_factorization(primes, exponents):
    """
    Calculate tau_perp given the prime factorization.
    primes: list of primes
    exponents: list of exponents
    """
    # Generate all divisors
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divisors = []
        power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divisors.append(d * power)
            power *= p
        divisors = new_divisors
    
    divisors.sort()
    
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            count += 1
            
    return count

def _score_key(primes, exponents):
    """Create a hashable key for memoization."""
    return (tuple(primes), tuple(exponents))

_score_cache = {}

def _score_cached(primes, exponents):
    """Score with caching."""
    key = _score_key(primes, exponents)
    if key in _score_cache:
        return _score_cache[key]
    result = _score_from_factorization(primes, exponents)
    _score_cache[key] = result
    return result

def construct():
    """
    Returns an integer n <= 10^18 maximizing tau_perp(n).
    """
    start_time = time.time()
    TIME_LIMIT = 145  # Safety margin for 150s kill
    
    max_val = 10**18
    
    # First 15 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 # Seed fallback
    
    # 1. Evaluate Maximal Primorial P15
    p15_val = 1
    for p in primes:
        p15_val *= p
    if p15_val <= max_val:
        score_p15 = _score_cached(primes, [1]*15)
        if score_p15 > best_score:
            best_score = score_p15
            best_n = p15_val
            
    # 2. Evaluate variations of P15
    # Strategy: Remove the largest prime(s) and increase exponents of smaller primes
    base_primes = primes[:14]
    base_val = 1
    for p in base_primes:
        base_val *= p
        
    candidates = []
    
    def add_candidate(prime_list, exps):
        val = 1
        for p, e in zip(prime_list, exps):
            val *= (p ** e)
        if val <= max_val:
            candidates.append((prime_list, exps, val))
            
    # 1. Single prime exponent increases on 14-prime base
    for i in range(5):  # First 5 primes
        for e in range(2, 10):
            exps = [1]*14
            exps[i] = e
            add_candidate(base_primes, exps)
            
    # 2. Two prime exponent increases
    for i in range(4):
        for j in range(i+1, 5):
            for ei in range(2, 6):
                for ej in range(2, 6):
                    exps = [1]*14
                    exps[i] = ei
                    exps[j] = ej
                    add_candidate(base_primes, exps)
                    
    # 3. Three prime exponent increases
    for i in range(3):
        for j in range(i+1, 4):
            for k in range(j+1, 5):
                for ei in range(2, 4):
                    for ej in range(2, 4):
                        for ek in range(2, 4):
                            exps = [1]*14
                            exps[i] = ei
                            exps[j] = ej
                            exps[k] = ek
                            add_candidate(base_primes, exps)
    
    # 4. 13-prime base with higher exponents
    base_primes_13 = primes[:13]
    for i in range(4):
        for e in range(2, 12):
            exps = [1]*13
            exps[i] = e
            add_candidate(base_primes_13, exps)
            
    for i in range(3):
        for j in range(i+1, 4):
            for ei in range(2, 8):
                for ej in range(2, 8):
                    exps = [1]*13
                    exps[i] = ei
                    exps[j] = ej
                    add_candidate(base_primes_13, exps)
    
    # 5. 12-prime base with even higher exponents
    base_primes_12 = primes[:12]
    for i in range(4):
        for e in range(2, 15):
            exps = [1]*12
            exps[i] = e
            add_candidate(base_primes_12, exps)
            
    for i in range(3):
        for j in range(i+1, 4):
            for ei in range(2, 10):
                for ej in range(2, 10):
                    exps = [1]*12
                    exps[i] = ei
                    exps[j] = ej
                    add_candidate(base_primes_12, exps)
    
    # Evaluate all candidates
    for p_list, e_list, val in candidates:
        if time.time() - start_time > TIME_LIMIT:
            break
        score = _score_cached(p_list, e_list)
        if score > best_score:
            best_score = score
            best_n = val
            
    # 3. Local search around the best candidate
    # Try small perturbations of the best factorization
    if time.time() - start_time < TIME_LIMIT - 30:
        # Get the factorization of best_n
        def factorize(n):
            factors = []
            temp = n
            for p in primes:
                if p * p > temp:
                    break
                count = 0
                while temp % p == 0:
                    count += 1
                    temp //= p
                if count > 0:
                    factors.append((p, count))
            if temp > 1:
                factors.append((temp, 1))
            return factors
        
        best_factors = factorize(best_n)
        best_p_list = [f[0] for f in best_factors]
        best_e_list = [f[1] for f in best_factors]
        
        # Try local perturbations
        local_candidates = set()
        
        # Try removing a prime
        for i in range(len(best_p_list)):
            new_p = best_p_list[:i] + best_p_list[i+1:]
            new_e = best_e_list[:i] + best_e_list[i+1:]
            val = 1
            for p, e in zip(new_p, new_e):
                val *= p ** e
            if val <= max_val:
                local_candidates.add((tuple(new_p), tuple(new_e), val))
        
        # Try increasing an exponent
        for i in range(len(best_p_list)):
            for de in range(1, 5):
                new_e = best_e_list.copy()
                new_e[i] += de
                val = 1
                for p, e in zip(best_p_list, new_e):
                    val *= p ** e
                if val <= max_val:
                    local_candidates.add((tuple(best_p_list), tuple(new_e), val))
        
        # Try decreasing an exponent (if > 1)
        for i in range(len(best_p_list)):
            if best_e_list[i] > 1:
                for de in range(1, best_e_list[i]):
                    new_e = best_e_list.copy()
                    new_e[i] -= de
                    val = 1
                    for p, e in zip(best_p_list, new_e):
                        val *= p ** e
                    if val <= max_val:
                        local_candidates.add((tuple(best_p_list), tuple(new_e), val))
        
        # Try replacing a prime with the next unused prime
        used_primes = set(best_p_list)
        for i in range(len(best_p_list)):
            for new_p in primes:
                if new_p in used_primes:
                    continue
                if new_p > best_p_list[i]:
                    new_p_list = best_p_list.copy()
                    new_p_list[i] = new_p
                    # Keep exponents the same
                    val = 1
                    for p, e in zip(new_p_list, best_e_list):
                        val *= p ** e
                    if val <= max_val:
                        local_candidates.add((tuple(new_p_list), tuple(best_e_list), val))
        
        # Evaluate local candidates
        for p_tuple, e_tuple, val in local_candidates:
            if time.time() - start_time > TIME_LIMIT:
                break
            score = _score_cached(list(p_tuple), list(e_tuple))
            if score > best_score:
                best_score = score
                best_n = val
                
    # 4. Additional targeted search: try primorials P_k for various k
    # with optimal exponent distributions
    if time.time() - start_time < TIME_LIMIT - 20:
        for num_primes in range(10, 16):
            if num_primes > len(primes):
                break
            p_list = primes[:num_primes]
            base_product = 1
            for p in p_list:
                base_product *= p
            
            # Try different exponent patterns
            # Pattern: all 1s (primorial)
            exps = [1] * num_primes
            val = base_product
            if val <= max_val:
                score = _score_cached(p_list, exps)
                if score > best_score:
                    best_score = score
                    best_n = val
            
            # Pattern: first few have higher exponents
            for max_exp in range(2, 6):
                for first_high in range(1, min(4, num_primes)):
                    exps = [1] * num_primes
                    for i in range(first_high):
                        exps[i] = max_exp
                    val = 1
                    for p, e in zip(p_list, exps):
                        val *= p ** e
                    if val <= max_val:
                        score = _score_cached(p_list, exps)
                        if score > best_score:
                            best_score = score
                            best_n = val
    
    return best_n
