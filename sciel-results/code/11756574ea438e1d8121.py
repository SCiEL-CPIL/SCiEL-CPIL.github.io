<<<<<<< SEARCH
import numpy as np

def kissing_number11() -> np.ndarray:
       d = 11
       points = []
       for i in range(d):
           for j in range(i+1, d):
               for si in [1, -1]:
                   for sj in [1, -1]:
                       p = np.zeros(d, dtype=np.int64)
                       p[i] = si
                       p[j] = sj
                       points.append(p)
       return np.array(points)
=======
import numpy as np
import time
import sys
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 590  # safety margin before 630s kill
    
    # Generate all candidate points: pairs of coordinates with ±1
    candidates = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    candidates.append(p)
    
    n = len(candidates)
    
    # Compute squared norms and squared distances
    norms = np.array([np.dot(p, p) for p in candidates])
    
    # Two candidates conflict if their squared distance < max_norm
    # We'll build conflict graph incrementally
    # For B&B, we need: can we add candidate c to current set S?
    # Condition: dist(c, s) >= max_norm(S ∪ {c}) for all s in S
    
    # Precompute pairwise squared distances
    # dist[i,j] = ||p_i - p_j||^2
    # = ||p_i||^2 + ||p_j||^2 - 2*p_i·p_j
    dots = np.array([np.dot(candidates[i], candidates[j]) for i in range(n) for j in range(n)]).reshape(n, n)
    dists = norms[:, None] + norms[None, :] - 2 * dots
    
    # For a set S with max_norm M, candidate c can be added iff
    # dist(c, s) >= M' for all s in S, where M' = max(M, norm(c))
    # This is equivalent to: min_s dist(c,s) >= M'
    
    # B&B with greedy coloring bound
    # State: (available set, current selection, current max_norm)
    
    best_set = []
    best_size = 0
    
    # Greedy initial solution
    greedy_set = []
    greedy_max_norm = 0
    for i in range(n):
        can_add = True
        for s in greedy_set:
            if dists[i, s] < max(greedy_max_norm, norms[i]):
                can_add = False
                break
        if can_add:
            greedy_set.append(i)
            greedy_max_norm = max(greedy_max_norm, norms[i])
    
    best_set = greedy_set[:]
    best_size = len(best_set)
    
    # B&B search
    # We'll use a recursive approach with memoization of available sets
    # But n=132 is large, so we need good pruning
    
    # Alternative: iterative deepening or better heuristic ordering
    
    # Sort candidates by norm (ascending) to try to keep max_norm small
    order = np.argsort(norms)
    
    # Try multiple random orderings with greedy + local search
    import random
    random.seed(42)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Random permutation
        perm = list(range(n))
        random.shuffle(perm)
        
        current_set = []
        current_max_norm = 0
        
        for idx in perm:
            can_add = True
            for s in current_set:
                if dists[idx, s] < max(current_max_norm, norms[idx]):
                    can_add = False
                    break
            if can_add:
                current_set.append(idx)
                current_max_norm = max(current_max_norm, norms[idx])
        
        if len(current_set) > best_size:
            best_set = current_set[:]
            best_size = len(current_set)
        
        # Local search: try to add more points
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if i in current_set:
                    continue
                can_add = True
                for s in current_set:
                    if dists[i, s] < max(current_max_norm, norms[i]):
                        can_add = False
                        break
                if can_add:
                    current_set.append(i)
                    current_max_norm = max(current_max_norm, norms[i])
                    improved = True
                    if len(current_set) > best_size:
                        best_set = current_set[:]
                        best_size = len(current_set)
                    break
        
        # Try removing one point and adding two
        if time.time() > deadline:
            break
        
        for remove_idx in range(len(current_set)):
            if time.time() > deadline:
                break
            
            saved = current_set[remove_idx]
            saved_max = current_max_norm
            current_set.pop(remove_idx)
            
            # Recompute max norm
            if current_set:
                current_max_norm = max(norms[s] for s in current_set)
            else:
                current_max_norm = 0
            
            added = []
            for i in range(n):
                if i in current_set:
                    continue
                can_add = True
                for s in current_set:
                    if dists[i, s] < max(current_max_norm, norms[i]):
                        can_add = False
                        break
                if can_add:
                    current_set.append(i)
                    current_max_norm = max(current_max_norm, norms[i])
                    added.append(i)
            
            if len(added) >= 2:
                if len(current_set) > best_size:
                    best_set = current_set[:]
                    best_size = len(current_set)
            
            # Restore
            current_set.remove(saved)
            current_max_norm = saved_max
    
    # Final local search on best
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if i in best_set:
                    continue
                can_add = True
                for s in best_set:
                    if dists[i, s] < max(max(norms[s] for s in best_set), norms[i]):
                        can_add = False
                        break
                if can_add:
                    best_set.append(i)
                    improved = True
                    break
    
    return np.array([candidates[i] for i in best_set])
>>>>>>> REPLACE
