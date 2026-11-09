import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin (90s limit, leave buffer)
    
    # Safe primes > 20: any product containing them is unique
    safe_primes = [23, 29, 31, 37]
    other_candidates = [x for x in range(2, 41) if x not in safe_primes]
    
    # Sort: primes first, then composites by number of distinct prime factors
    def score_candidate(x):
        n = x
        factors = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors += 1
        return (1, factors)
    
    other_candidates.sort(key=score_candidate)
    
    best_set = list(safe_primes)
    best_size = len(best_set)
    
    def build_products(S):
        P = {1}
        for x in S:
            P = P | {p * x for p in P}
        return P
    
    P_safe = build_products(safe_primes)
    
    def is_valid(P, x):
        for p in P:
            if p * x in P:
                return False
        return True
    
    def dfs(index, S, P):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
        
        if index == len(other_candidates):
            return
        
        remaining = len(other_candidates) - index
        if len(S) + remaining <= best_size:
            return
        
        x = other_candidates[index]
        
        # Option 1: Include x
        if is_valid(P, x):
            new_P = P | {p * x for p in P}
            dfs(index + 1, S + [x], new_P)
        
        # Option 2: Exclude x
        dfs(index + 1, S, P)
    
    # First pass: standard DFS
    dfs(0, list(safe_primes), P_safe)
    
    # Second pass: try reverse ordering if time permits
    if time.time() < deadline - 15:
        other_candidates_rev = list(reversed(other_candidates))
        
        def dfs_rev(index, S, P):
            nonlocal best_set, best_size
            if time.time() > deadline:
                return
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = list(S)
            
            if index == len(other_candidates_rev):
                return
            
            remaining = len(other_candidates_rev) - index
            if len(S) + remaining <= best_size:
                return
            
            x = other_candidates_rev[index]
            
            if is_valid(P, x):
                new_P = P | {p * x for p in P}
                dfs_rev(index + 1, S + [x], new_P)
            
            dfs_rev(index + 1, S, P)
        
        dfs_rev(0, list(safe_primes), P_safe)
    
    # Final polish: greedily add any remaining candidates
    best_set_set = set(best_set)
    P_best = build_products(best_set)
    
    remaining_cands = [x for x in range(2, 41) if x not in best_set_set]
    remaining_cands.sort(key=score_candidate)
    
    for x in remaining_cands:
        if time.time() > deadline:
            break
        if is_valid(P_best, x):
            P_best = P_best | {p * x for p in P_best}
            best_set.append(x)
    
    return [int(x) for x in best_set]
