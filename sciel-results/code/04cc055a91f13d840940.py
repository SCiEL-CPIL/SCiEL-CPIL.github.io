"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

MAX_SUM = 20000
_pf_cache = [set() for _ in range(MAX_SUM + 1)]

def _init_prime_factors():
    for i in range(2, MAX_SUM + 1):
        factors = set()
        n = i
        f = 2
        while f * f <= n:
            while n % f == 0:
                factors.add(f)
                n //= f
            f += 1
        if n > 1:
            factors.add(n)
        _pf_cache[i] = factors

_init_prime_factors()

def get_initial_candidates():
    candidates = []
    n = 12
    max_val = 10000
    
    # Strategy 1: Avoid 3, 5, 7
    # x_i = 105*k + 1
    mod1 = 105
    res1 = 1
    cand1 = [res1 + mod1 * k for k in range(n)]
    if max(cand1) <= max_val:
        candidates.append(cand1)
        
    # Strategy 2: Avoid 3, 5, 11
    # x_i = 165*k + 1
    mod2 = 165
    res2 = 1
    cand2 = [res2 + mod2 * k for k in range(n)]
    if max(cand2) <= max_val:
        candidates.append(cand2)
        
    # Strategy 3: Avoid 3, 7, 11
    # x_i = 231*k + 1
    mod3 = 231
    res3 = 1
    cand3 = [res3 + mod3 * k for k in range(n)]
    if max(cand3) <= max_val:
        candidates.append(cand3)
        
    # Strategy 4: Avoid 5, 7, 11
    # x_i = 385*k + 1
    mod4 = 385
    res4 = 1
    cand4 = [res4 + mod4 * k for k in range(n)]
    if max(cand4) <= max_val:
        candidates.append(cand4)

    # Strategy 5: Avoid 3, 5
    # x_i = 15*k + 1
    mod5 = 15
    res5 = 1
    cand5 = [res5 + mod5 * k for k in range(n)]
    if max(cand5) <= max_val:
        candidates.append(cand5)
        
    # Add a random odd set as fallback
    random.seed(42)
    cand_rand = random.sample(range(1, max_val+1, 2), n)
    candidates.append(cand_rand)
    
    return candidates

def count_distinct_primes(nums):
    primes = set()
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            primes |= _pf_cache[nums[i] + nums[j]]
    return len(primes)

def construct():
    n = 12
    max_val = 10000
    
    candidates = get_initial_candidates()
    
    best = None
    best_score = float('inf')
    
    for init in candidates:
        if len(set(init)) != n or min(init) < 1 or max(init) > max_val:
            continue
        score = count_distinct_primes(init)
        if score < best_score:
            best_score = score
            best = init[:]
            
    if best is None:
        return list(range(1, n+1))
        
    current = best[:]
    current_score = best_score
    
    # Fast Local Search
    # Maintain prime counts
    prime_counts = {}
    def update_prime_counts(nums, delta):
        # delta is +1 for adding a set of sums, -1 for removing
        # This function is tricky to implement generically for a single element swap
        # Simpler: Just recalculate if n is small? n=12, pairs=66.
        # 66 set unions is fast.
        # Let's stick to full recalculation but optimize the loop.
        pass

    # Actually, for n=12, 66 pairs.
    # Set union of small sets is very fast in Python.
    # The bottleneck was the 30000 iterations.
    # Let's try 5000 iterations of local search.
    
    random.seed(123)
    for it in range(5000):
        i = random.randint(0, n-1)
        # Try to swap with a value that might be better
        # Heuristic: try values close to current or from the initial pool
        # For now, random replacement
        new_val = random.randint(1, max_val)
        if new_val in current:
            continue
        
        old_val = current[i]
        current[i] = new_val
        new_score = count_distinct_primes(current)
        
        if new_score <= current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = current[:]
        else:
            current[i] = old_val
            
    return best

# EVOLVE-BLOCK-END
