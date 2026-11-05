import time
import random
import numpy as np

def set_sizes(A):
    A = list(A)
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return len(sums), len(prods)

def objective(A):
    s, p = set_sizes(A)
    return max(s, p)

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s
    
    # 1. Generate diverse structural seeds
    seeds = []
    
    # Seed 1: Arithmetic Progression
    seeds.append(list(range(1, 21)))
    
    # Seed 2: Geometric Progression (base 2)
    seeds.append([2**i for i in range(20)])
    
    # Seed 3: Geometric Progression (base 1.5, rounded)
    gp = [int(round(1.5 ** i)) for i in range(20)]
    # Ensure distinct
    seen = set()
    uniq = []
    for x in gp:
        if x not in seen and 1 <= x <= 10000:
            seen.add(x)
            uniq.append(x)
    while len(uniq) < 20:
        uniq.append(uniq[-1] + 1)
    seeds.append(uniq[:20])
    
    # Seed 4: Hybrid (AP for small, GP for large)
    hybrid = list(range(1, 11))
    for i in range(10):
        hybrid.append(int(10 * (2 ** i)))
    # Deduplicate and fix
    seen = set()
    uniq = []
    for x in hybrid:
        if x not in seen and 1 <= x <= 10000:
            seen.add(x)
            uniq.append(x)
    while len(uniq) < 20:
        uniq.append(uniq[-1] + 1)
    seeds.append(uniq[:20])

    # Seed 5: Random geometric with optimization
    for r in [1.2, 1.3, 1.4, 1.6, 1.8]:
        gp = [int(round(r ** i)) for i in range(20)]
        seen = set()
        uniq = []
        for x in gp:
            if x not in seen and 1 <= x <= 10000:
                seen.add(x)
                uniq.append(x)
        while len(uniq) < 20:
            uniq.append(uniq[-1] + 1)
        seeds.append(uniq[:20])

    # Evaluate seeds
    best_set = None
    best_obj = float('inf')
    for s in seeds:
        if len(set(s)) != 20 or any(x < 1 or x > 10000 for x in s):
            continue
        obj = objective(s)
        if obj < best_obj:
            best_obj = obj
            best_set = s[:]

    # 2. Local Search with improved neighborhood
    # We will perform a greedy descent from multiple starting points.
    
    def local_search(A_init):
        nonlocal best_set, best_obj
        A = list(A_init)
        cur_obj = objective(A)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_set = A[:]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to improve each position
            for i in range(20):
                if time.time() > deadline - 5:
                    break
                old_val = A[i]
                # Generate candidates for position i
                # Strategy: Try values that are close to old_val, or multiples/fractions
                candidates = set()
                # Local additive
                for d in range(-5, 6):
                    candidates.add(old_val + d)
                # Local multiplicative
                for factor in [0.8, 0.9, 1.1, 1.2, 1.5, 2.0]:
                    candidates.add(int(old_val * factor))
                # Random global (to escape local minima)
                candidates.add(random.randint(1, 10000))
                
                best_candidate_val = old_val
                best_candidate_obj = cur_obj
                
                for cand in candidates:
                    if cand < 1 or cand > 10000 or cand in A:
                        continue
                    A[i] = cand
                    new_obj = objective(A)
                    if new_obj < best_candidate_obj:
                        best_candidate_obj = new_obj
                        best_candidate_val = cand
                    A[i] = old_val
                
                if best_candidate_val != old_val:
                    A[i] = best_candidate_val
                    cur_obj = best_candidate_obj
                    improved = True
                    if cur_obj < best_obj:
                        best_obj = cur_obj
                        best_set = A[:]
        
        return A

    # Run local search from the best seed and a few others
    random.seed(42)
    
    # Run on best seed
    if time.time() < deadline - 10:
        local_search(best_set)
        
    # Run on other seeds if time permits
    for s in seeds:
        if time.time() > deadline - 10:
            break
        # Only run if this seed wasn't the best one already explored
        if s == best_set:
            continue
        local_search(s)
        
    # Final polish from the global best
    if time.time() < deadline - 2:
        local_search(best_set)

    return best_set
