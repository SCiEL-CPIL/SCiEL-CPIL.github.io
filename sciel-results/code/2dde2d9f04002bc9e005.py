import math
import time
from functools import lru_cache

@lru_cache(maxsize=None)
def _score_from_factorization(primes_tuple, exponents_tuple):
    """
    Calculate tau_perp given the prime factorization.
    Uses bitmask representation for efficient coprime checking.
    """
    primes = list(primes_tuple)
    exponents = list(exponents_tuple)
    
    if not primes:
        return 0
    
    num_primes = len(primes)
    
    # Generate all divisors with their prime masks
    # Each divisor is represented as (value, mask) where mask indicates which primes divide it
    divisors = [(1, 0)]
    
    for idx, (p, e) in enumerate(zip(primes, exponents)):
        new_divisors = []
        bit = 1 << idx
        power = 1
        for _ in range(e + 1):
            for val, mask in divisors:
                if power == 1:
                    new_divisors.append((val, mask))
                else:
                    new_divisors.append((val * power, mask | bit))
            power *= p
        divisors = new_divisors
    
    # Sort by value
    divisors.sort(key=lambda x: x[0])
    
    count = 0
    for i in range(len(divisors) - 1):
        if divisors[i][1] & divisors[i+1][1] == 0:
            count += 1
            
    return count

def _score_cached(primes, exponents):
    """Score with caching."""
    return _score_from_factorization(tuple(primes), tuple(exponents))

def construct():
    """
    Returns an integer n <= 10^18 maximizing tau_perp(n).
    """
    start_time = time.time()
    TIME_LIMIT = 140  # Safety margin for 150s kill
    
    max_val = 10**18
    
    # First 15 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 # Seed fallback
    
    # 1. Evaluate Maximal Primorial P15 as a strong baseline
    p15_val = 1
    for p in primes:
        p15_val *= p
    if p15_val <= max_val:
        score_p15 = _score_cached(primes, [1]*15)
        if score_p15 > best_score:
            best_score = score_p15
            best_n = p15_val
            
    # 2. DFS to search for better candidates with non-increasing exponents
    # State: idx (prime index), max_exp (max exponent for this prime), current_n, current_primes, current_exps
    def dfs(idx, max_exp, current_n, current_primes, current_exps):
        nonlocal best_score, best_n
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        # Evaluate current state if it has at least one prime factor
        if len(current_primes) > 0:
            score = _score_cached(current_primes, current_exps)
            if score > best_score:
                best_score = score
                best_n = current_n

        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        # Try exponents from max_exp down to 1
        # We can stop if current_n * p^e > max_val
        # Note: We enforce non-increasing exponents: e <= max_exp
        for e in range(max_exp, 0, -1):
            new_n = current_n * (p ** e)
            if new_n > max_val:
                continue
                
            # Recurse
            # Next prime is idx + 1
            # Max exponent for next prime is e (non-increasing constraint)
            dfs(idx + 1, e, new_n, current_primes + [p], current_exps + [e])
            
            # Early termination for time
            if time.time() - start_time > TIME_LIMIT:
                return

    # Start DFS
    # Initial max_exp can be high, e.g., 60 (since 2^60 > 1e18)
    # But we can bound it tighter: floor(log_2(max_val))
    max_initial_exp = int(math.log2(max_val))
    dfs(0, max_initial_exp, 1, [], [])
    
    # 3. Local search around the best candidate to catch any non-smooth anomalies
    # If the optimal solution violated the non-increasing exponent assumption slightly,
    # local search can fix it.
    if time.time() - start_time < TIME_LIMIT - 10:
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
        
        # Try small perturbations
        for i in range(len(best_p_list)):
            for de in [-2, -1, 1, 2, 3]:
                new_e = best_e_list.copy()
                new_e[i] += de
                if new_e[i] < 1:
                    continue
                val = 1
                for p, e in zip(best_p_list, new_e):
                    val *= p ** e
                if val <= max_val:
                    score = _score_cached(best_p_list, new_e)
                    if score > best_score:
                        best_score = score
                        best_n = val
                        
        # Try swapping primes (replace a prime with another unused one)
        used_primes = set(best_p_list)
        for i in range(len(best_p_list)):
            for new_p in primes:
                if new_p in used_primes:
                    continue
                new_p_list = best_p_list.copy()
                new_p_list[i] = new_p
                val = 1
                for p, e in zip(new_p_list, best_e_list):
                    val *= p ** e
                if val <= max_val:
                    score = _score_cached(new_p_list, best_e_list)
                    if score > best_score:
                        best_score = score
                        best_n = val
                        
        # Try adding a new prime (if we have room and it's small)
        if len(best_p_list) < 16:
            for new_p in primes[len(best_p_list):]:
                new_p_list = best_p_list + [new_p]
                new_e_list = best_e_list + [1]
                val = 1
                for p, e in zip(new_p_list, new_e_list):
                    val *= p ** e
                if val <= max_val:
                    score = _score_cached(new_p_list, new_e_list)
                    if score > best_score:
                        best_score = score
                        best_n = val
                        
    return best_n
