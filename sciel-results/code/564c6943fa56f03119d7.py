import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    N = 511
    
    # 1. Precompute prime factors for all numbers 1..511
    factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not factors[i]:  # i is prime
            for j in range(i, N + 1, i):
                factors[j].add(i)
    
    # 2. Identify unique prime factor sets and their best representatives
    # Map: frozenset(prime_factors) -> max_a
    best_representative = {}
    for a in range(1, N + 1):
        fac_set = frozenset(factors[a])
        if fac_set not in best_representative:
            best_representative[fac_set] = a
        else:
            if a > best_representative[fac_set]:
                best_representative[fac_set] = a
    
    # 3. Prepare the reduced list of candidates
    # Each candidate is (a, frozenset_factors, weight)
    candidates = []
    for fac_set, a in best_representative.items():
        weight = 1.0 / (512 - a)
        candidates.append((a, fac_set, weight))
    
    # Sort by weight descending to help B&B find good solutions early
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Convert frozensets to bitmasks for fast intersection checks
    # We need a mapping from prime to bit index
    # Since we only care about primes < 512, we can map them to 0..96
    primes = [p for p in range(2, N + 1) if not factors[p]] # Wait, factors[p] is empty for primes?
    # My factor loop: 
    # for i in 2..511: if not factors[i]: factors[i] is empty.
    # Then for j in i..511 step i: factors[j].add(i).
    # So for prime p, factors[p] will contain p.
    # So `if not factors[p]` is false for primes.
    # Let's re-identify primes properly.
    primes = []
    for p in range(2, N + 1):
        is_prime = True
        for q in primes:
            if q * q > p:
                break
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
            
    prime_index = {p: i for i, p in enumerate(primes)}
    
    candidate_bitmasks = []
    for a, fac_set, weight in candidates:
        mask = 0
        for p in fac_set:
            mask |= (1 << prime_index[p])
        candidate_bitmasks.append(mask)
    
    M = len(candidates)
    
    # 4. Branch and Bound
    best_score = 0.0
    best_solution = []
    
    # Precompute suffix sums of weights for a loose upper bound?
    # No, the "sum of compatible weights" bound is better and dynamic.
    
    # To speed up the bound calculation, we can precompute nothing specific,
    # but the loop over M is small (likely < 100 after reduction).
    
    def dfs(idx, current_mask, current_score, current_set):
        nonlocal best_score, best_solution
        
        if time.time() > deadline:
            return
            
        # Upper Bound: Sum of weights of all remaining candidates compatible with current_mask
        ub = current_score
        for j in range(idx, M):
            if (candidate_bitmasks[j] & current_mask) == 0:
                ub += candidates[j][2]
                
        if ub <= best_score:
            return
            
        if idx == M:
            if current_score > best_score:
                best_score = current_score
                best_solution = current_set.copy()
            return
            
        # Branch 1: Include candidate[idx] if possible
        a, fac_set, weight = candidates[idx]
        mask = candidate_bitmasks[idx]
        
        if (mask & current_mask) == 0:
            current_set.append(a)
            dfs(idx + 1, current_mask | mask, current_score + weight, current_set)
            current_set.pop()
            
        # Branch 2: Exclude candidate[idx]
        dfs(idx + 1, current_mask, current_score, current_set)
        
    dfs(0, 0, 0.0, [])
    
    return best_solution
