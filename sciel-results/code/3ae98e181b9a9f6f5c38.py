import time
import math
import sys

def construct():
    N = 511
    deadline = time.time() + 110.0 # Safety margin
    
    # 1. Precompute weights and sort
    # Values are 1..511
    # Weight is 1/(512 - a)
    # We want to maximize sum, so larger a (closer to 511) have larger weight.
    # Sort by weight descending means sorting by a descending.
    values = list(range(511, 0, -1))
    w = [1.0 / (512 - v) for v in values]
    
    # 2. Precompute Conflict Masks
    # conflict_mask[v] is a bitmask of all u in [1, 511] such that gcd(v, u) > 1
    # Bit u corresponds to value u.
    # We use Python ints which can be arbitrarily large.
    
    # Sieve for smallest prime factor (SPF) to factorize quickly
    spf = list(range(N + 1))
    for i in range(2, int(N ** 0.5) + 1):
        if spf[i] == i: # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    def get_prime_factors(n):
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    conflict_mask = [0] * (N + 1)
    for v in range(1, N + 1):
        pf = get_prime_factors(v)
        mask = 0
        for p in pf:
            for u in range(p, N + 1, p):
                mask |= (1 << u)
        conflict_mask[v] = mask
        
    # 3. Greedy Initial Solution
    # This provides a strong lower bound (initial best_weight)
    def greedy_init():
        active_mask = (1 << (N + 1)) - 1 # All bits 1..N set
        # Actually, bit 0 is unused.
        active_mask = 0
        for i in range(1, N + 1):
            active_mask |= (1 << i)
            
        current_weight = 0.0
        current_set = []
        
        for i in range(len(values)):
            v = values[i]
            if (active_mask >> v) & 1:
                # v is compatible
                current_set.append(v)
                current_weight += w[i]
                active_mask &= ~conflict_mask[v]
                
        return current_weight, current_set

    best_weight, best_set = greedy_init()
    
    # 4. Branch and Bound
    sys.setrecursionlimit(10000)
    
    def dfs(idx, active_mask, current_weight, current_set):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return
            
        # Upper Bound
        # Sum of weights of all compatible remaining vertices
        ub = current_weight
        for j in range(idx, len(values)):
            v = values[j]
            if (active_mask >> v) & 1:
                ub += w[j]
                
        if ub <= best_weight:
            return
            
        if idx == len(values):
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = current_set[:]
            return
        
        v = values[idx]
        
        # Branch 1: Include v (if possible)
        if (active_mask >> v) & 1:
            new_mask = active_mask & ~conflict_mask[v]
            current_set.append(v)
            dfs(idx + 1, new_mask, current_weight + w[idx], current_set)
            current_set.pop()
            
        # Branch 2: Exclude v
        dfs(idx + 1, active_mask, current_weight, current_set)
        
    # Start B&B
    # Initial mask: all bits 1..511 set
    initial_mask = 0
    for i in range(1, N + 1):
        initial_mask |= (1 << i)
        
    dfs(0, initial_mask, 0.0, [])
    
    return best_set
