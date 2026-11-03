import numpy as np
import itertools
import time

def construct():
    D = 11
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Generate candidate vectors: all {-1,0,1} vectors with norm^2 in {1, 2, 3}
    candidates = []
    
    # Norm^2 = 1: single nonzero entry
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int8)
            v[i] = s
            candidates.append(v)
    
    # Norm^2 = 2: two nonzero entries
    for support in itertools.combinations(range(D), 2):
        i, j = support
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(D, dtype=np.int8)
                v[i] = s1
                v[j] = s2
                candidates.append(v)
    
    # Norm^2 = 3: three nonzero entries
    for support in itertools.combinations(range(D), 3):
        i, j, k = support
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = s1
                    v[j] = s2
                    v[k] = s3
                    candidates.append(v)
    
    n = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.array([np.dot(v, v) for v in candidates], dtype=np.int32)
    
    # Precompute inner products
    cand_arr = np.array(candidates, dtype=np.int8)
    inner_products = cand_arr @ cand_arr.T
    
    # Conflict matrix
    conflict = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            min_norm_sq = min(norms_sq[i], norms_sq[j])
            ip = inner_products[i, j]
            if 2 * ip > min_norm_sq:
                conflict[i, j] = True
                conflict[j, i] = True
    
    # Greedy initial solution: repeatedly pick vertex with min degree in remaining graph
    greedy_set = []
    remaining = set(range(n))
    while remaining:
        # Find vertex with minimum number of conflicts in remaining
        min_deg = float('inf')
        chosen = -1
        for v in remaining:
            deg = sum(1 for u in remaining if u != v and conflict[v, u])
            if deg < min_deg:
                min_deg = deg
                chosen = v
        if chosen == -1:
            break
        greedy_set.append(chosen)
        # Remove chosen and all its neighbors
        remaining.discard(chosen)
        for u in list(remaining):
            if conflict[chosen, u]:
                remaining.discard(u)
    
    best_set = list(greedy_set)
    best_size = len(best_set)
    
    # Backtracking with MRV and pruning
    def backtrack(available, current_set):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        
        # Pruning: even if we add all available, can we beat best?
        if len(current_set) + len(available) <= best_size:
            return
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
        
        if not available:
            return
        
        avail_list = list(available)
        
        # MRV: pick vertex with max degree in available subgraph
        max_deg = -1
        chosen = -1
        for v in avail_list:
            deg = sum(1 for u in avail_list if u != v and conflict[v, u])
            if deg > max_deg:
                max_deg = deg
                chosen = v
        
        if chosen == -1:
            return
        
        # Branch 1: include chosen
        new_available = set()
        for u in available:
            if u != chosen and not conflict[chosen, u]:
                new_available.add(u)
        current_set.append(chosen)
        backtrack(new_available, current_set)
        current_set.pop()
        
        if time.time() > deadline:
            return
        
        # Branch 2: exclude chosen
        new_available2 = set(available)
        new_available2.discard(chosen)
        backtrack(new_available2, current_set)
    
    initial_available = set(range(n))
    backtrack(initial_available, [])
    
    # Build result
    if best_set:
        result = np.array([candidates[i] for i in best_set], dtype=np.int8)
    else:
        vectors = []
        for support in itertools.combinations(range(D), 2):
            i, j = support
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = s1
                    v[j] = s2
                    vectors.append(v)
        result = np.array(vectors, dtype=np.int8)
    
    return result
