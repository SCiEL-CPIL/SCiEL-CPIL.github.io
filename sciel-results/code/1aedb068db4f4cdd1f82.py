import time
import sys

def _prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Safe primes: primes > 20 that can't cause collisions
    safe_primes = [23, 29, 31, 37]
    other_candidates = [x for x in range(2, 41) if x not in safe_primes]
    
    best_set = list(safe_primes)
    best_size = len(best_set)
    
    # Build P_base for safe primes
    P_base = {1}
    for x in safe_primes:
        new_P = set(P_base)
        for p in P_base:
            new_P.add(p * x)
        P_base = new_P
    
    # Precompute prime factors
    pf_cache = {x: _prime_factors(x) for x in range(2, 41)}
    
    # Conflict matrix: which pairs of candidates can't be together
    # Two numbers conflict if their product or any combination causes collision
    def is_valid(P, x):
        for p in P:
            if p * x in P:
                return False
        return True
    
    def dfs(index, S, P, candidates, time_limit):
        nonlocal best_set, best_size
        if time.time() > time_limit:
            return
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
        
        if index == len(candidates):
            return
        
        # Pruning
        if len(S) + (len(candidates) - index) <= best_size:
            return
        
        x = candidates[index]
        
        # Try including x
        if is_valid(P, x):
            new_P = set(P)
            for p in P:
                new_P.add(p * x)
            dfs(index + 1, S + [x], new_P, candidates, time_limit)
        
        # Try excluding x
        dfs(index + 1, S, P, candidates, time_limit)
    
    # Generate orderings based on different heuristics
    orderings = []
    
    # Ordering 1: Descending value (larger numbers less likely to conflict)
    orderings.append(sorted(other_candidates, reverse=True))
    
    # Ordering 2: By number of distinct prime factors (fewer factors = less conflict potential)
    orderings.append(sorted(other_candidates, key=lambda x: (len(pf_cache[x]), -x)))
    
    # Ordering 3: By number of distinct prime factors ascending, then value
    orderings.append(sorted(other_candidates, key=lambda x: (len(pf_cache[x]), x)))
    
    # Ordering 4: Primes first, then composites
    primes_in_range = [x for x in other_candidates if len(pf_cache[x]) == 1]
    composites_in_range = [x for x in other_candidates if len(pf_cache[x]) > 1]
    orderings.append(primes_in_range + composites_in_range)
    
    # Ordering 5: By "conflict potential" - numbers sharing prime factors with many others
    def conflict_score(x):
        score = 0
        for y in other_candidates:
            if y != x and pf_cache[x] & pf_cache[y]:
                score += 1
        return score
    orderings.append(sorted(other_candidates, key=lambda x: -conflict_score(x)))
    
    # Ordering 6: Ascending value
    orderings.append(sorted(other_candidates))
    
    # Allocate time: 8 seconds per ordering, total ~48s for 6 orderings
    n_orderings = len(orderings)
    time_per_ordering = 8
    
    for i, ordering in enumerate(orderings):
        if time.time() > deadline - 5:
            break
        t_limit = min(start_time + time_per_ordering * (i + 1) + 2, deadline - 5)
        dfs(0, list(safe_primes), P_base, ordering, t_limit)
    
    # Final polish: try to add remaining elements
    best_set_set = set(best_set)
    remaining = [x for x in range(2, 41) if x not in best_set_set]
    
    P_best = {1}
    for x in best_set:
        new_P = set(P_best)
        for p in P_best:
            new_P.add(p * x)
        P_best = new_P
    
    # Try adding in different orders
    for order in [sorted(remaining), sorted(remaining, reverse=True)]:
        P_temp = set(P_best)
        S_temp = list(best_set)
        for x in order:
            if time.time() > deadline:
                break
            if is_valid(P_temp, x):
                new_P = set(P_temp)
                for p in P_temp:
                    new_P.add(p * x)
                P_temp = new_P
                S_temp.append(x)
        
        if len(S_temp) > best_size:
            best_size = len(S_temp)
            best_set = S_temp
            P_best = P_temp
    
    # Try local search: swap elements to find better solutions
    best_set_set = set(best_set)
    remaining = [x for x in range(2, 41) if x not in best_set_set]
    
    P_best = {1}
    for x in best_set:
        new_P = set(P_best)
        for p in P_best:
            new_P.add(p * x)
        P_best = new_P
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i, elem in enumerate(list(best_set)):
            if time.time() > deadline - 2:
                break
            for new_elem in remaining:
                if time.time() > deadline - 2:
                    break
                new_S = list(best_set)
                new_S[i] = new_elem
                new_P = {1}
                valid = True
                for x in new_S:
                    new_P_temp = set(new_P)
                    for p in new_P:
                        new_P_temp.add(p * x)
                    new_P = new_P_temp
                
                if len(new_P) == 2 ** len(new_S):
                    if len(new_S) > best_size:
                        best_size = len(new_S)
                        best_set = new_S
                        P_best = new_P
                        improved = True
                        break
    
    return [int(x) for x in best_set]
