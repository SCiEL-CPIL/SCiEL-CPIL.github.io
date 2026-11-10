import time
import math
import random

def construct():
    start_time = time.time()
    LIMIT = 10**18
    DEADLINE = start_time + 115.0  # Safety margin for 120s limit

    # Generate primes up to 1000
    def gen_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = gen_primes(1000)
    
    # Ensure we have at least 15 primes
    if len(primes) < 15:
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    def tau_perp_score(prime_list):
        """
        Calculate tau_perp for a square-free number given its list of prime factors.
        """
        k = len(prime_list)
        # Generate divisors using bitmasks
        divs = []
        for mask in range(1 << k):
            val = 1
            for i in range(k):
                if mask & (1 << i):
                    val *= prime_list[i]
            divs.append(val)
        
        divs.sort()
        
        score = 0
        prev = divs[0]
        for curr in divs[1:]:
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
        return score

    def get_product(p_list):
        p = 1
        for x in p_list:
            p *= x
        return p

    def random_valid_set():
        # Strategy: Start with a random selection, then fix product constraint
        # Pick 15 distinct random primes from the pool.
        # If product > LIMIT, replace largest primes with smallest available primes until valid.
        
        available = primes.copy()
        chosen = []
        for _ in range(15):
            idx = random.randint(0, len(available)-1)
            chosen.append(available[idx])
            available.pop(idx)
        
        chosen.sort()
        prod = get_product(chosen)
        
        # Fix product if too large
        # Use small primes for replacement
        small_primes = primes[:50]
        
        while prod > LIMIT:
            # Find the largest prime in chosen
            max_idx = chosen.index(max(chosen))
            max_p = chosen[max_idx]
            
            # Find the smallest prime in small_primes that is not in chosen
            replacement = None
            for p in small_primes:
                if p not in chosen:
                    replacement = p
                    break
            
            if replacement is None or replacement >= max_p:
                # Can't reduce further, return default
                return primes[:15].copy()
            
            # Perform swap
            prod = prod // max_p * replacement
            chosen[max_idx] = replacement
            chosen.sort()
            
        return chosen

    best_n = 0
    best_score = -1
    score_cache = {}

    def get_cached_score(p_list):
        key = tuple(sorted(p_list))
        if key in score_cache:
            return score_cache[key]
        s = tau_perp_score(p_list)
        score_cache[key] = s
        return s

    # Initialize with a few random restarts
    for _ in range(20):
        if time.time() > DEADLINE - 10: break
        set_ = random_valid_set()
        score = get_cached_score(set_)
        if score > best_score:
            best_score = score
            best_n = get_product(set_)
    
    # Local Search with Random Restarts
    max_restarts = 100
    for r in range(max_restarts):
        if time.time() > DEADLINE - 5: break
        
        current_set = random_valid_set()
        current_score = get_cached_score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_n = get_product(current_set)
            
        stall = 0
        max_stall = 200 # If no improvement in 200 moves, restart
        
        while stall < max_stall and time.time() < DEADLINE - 2:
            # Generate neighbor
            idx = random.randint(0, 14)
            p_old = current_set[idx]
            prod = get_product(current_set)
            max_p_new = LIMIT * p_old // prod
            
            # Candidates: primes not in current_set, within [2, max_p_new]
            candidates = [p for p in primes if p not in current_set and p <= max_p_new]
            if not candidates:
                stall += 1
                continue
            
            p_new = random.choice(candidates)
            
            new_set = current_set.copy()
            new_set[idx] = p_new
            new_set.sort()
            
            new_prod = prod // p_old * p_new
            
            new_score = get_cached_score(new_set)
            
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                stall = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_n = new_prod
            else:
                stall += 1
                
    if best_n == 0:
        # Fallback to first 15 primes
        best_n = 1
        for p in primes[:15]:
            best_n *= p
            
    assert 2 <= best_n <= LIMIT
    return best_n
