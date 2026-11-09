import numpy as np
import time
from itertools import combinations, product
import random

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580  # safety margin
    
    # Fallback: 220-point solution (norm^2 = 2)
    # Vectors with exactly 2 non-zero entries (±1)
    candidates_2 = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates_2.append(v)
    
    fallback_points = np.array(candidates_2, dtype=np.int64)
    fallback_k = len(fallback_points) # 220
    
    # Try to find a better solution using vectors with exactly 3 non-zero entries (±1)
    # norm^2 = 3, need pairwise dist^2 >= 3, i.e., dot <= 1
    candidates = []
    for pos in combinations(range(d), 3):
        for signs in product([-1, 1], repeat=3):
            v = np.zeros(d, dtype=np.int64)
            for p, s in zip(pos, signs):
                v[p] = s
            candidates.append(v)
    
    n_cands = len(candidates) # 1320
    
    # Precompute conflict graph
    # Conflict if dot > 1
    # We want an independent set in the conflict graph
    adj = [set() for _ in range(n_cands)]
    
    # Build adjacency list of conflicts
    # Optimization: Compute dots only for i < j
    for i in range(n_cands):
        for j in range(i + 1, n_cands):
            dot = np.dot(candidates[i], candidates[j])
            if dot > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    best_k = 0
    best_selected = None
    
    # Strategy: Randomized Greedy with different ordering heuristics
    # We cycle through: Random, Ascending Index, Descending Index
    # This aligns with the "cell findings" that varying ordering helped.
    
    n_trials = 300
    random.seed(42)
    
    for trial in range(n_trials):
        if time.time() > deadline:
            break
            
        # Determine ordering strategy based on trial number
        # 0: Random, 1: Ascending, 2: Descending
        strategy = trial % 3
        
        if strategy == 0:
            order = list(range(n_cands))
            random.shuffle(order)
        elif strategy == 1:
            order = list(range(n_cands))
        else:
            order = list(range(n_cands - 1, -1, -1))
        
        selected = []
        selected_set = set()
        
        for idx in order:
            # Check if any neighbor is already selected
            # adj[idx] is a set of neighbors
            if not (adj[idx] & selected_set):
                selected.append(idx)
                selected_set.add(idx)
        
        k = len(selected)
        if k > best_k:
            best_k = k
            best_selected = selected
            # Optional: Early stop if we hit a theoretical bound? 
            # Max kissing number in 11D is unknown but > 500.
            # If we get close to a known high value, we might stop, but let's run all trials.

    if best_selected is not None and best_k > fallback_k:
        points = np.array([candidates[i] for i in best_selected], dtype=np.int64)
        
        # Verify with exact arithmetic
        k = points.shape[0]
        norms = np.sum(points**2, axis=1)
        max_norm = np.max(norms)
        
        valid = True
        # The check is O(k^2 * d). If k ~ 500, k^2 = 250k, d=11. ~2.7M ops. Fast.
        for i in range(k):
            if time.time() > deadline:
                break
            for j in range(i+1, k):
                diff = points[i] - points[j]
                dist_sq = np.sum(diff**2)
                if dist_sq < max_norm:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return points

    return fallback_points
