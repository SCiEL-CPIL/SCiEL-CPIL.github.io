import numpy as np
import time
import itertools
import random
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    # 1. Generate Baseline: Norm 2 vectors (220 points)
    candidates_2 = []
    for combo in itertools.combinations(range(d), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(d, dtype=np.int64)
            v[list(combo)] = signs
            candidates_2.append(v)
    
    base_solution = np.array(candidates_2, dtype=np.int64)
    best_points = base_solution
    best_k = len(base_solution)
    
    # 2. Generate Norm 4 vectors
    candidates_4 = []
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            v[list(combo)] = signs
            candidates_4.append(v)
            
    if not candidates_4:
        return best_points
        
    V4 = np.array(candidates_4, dtype=np.int64)
    N4 = len(V4)
    
    # 3. Build Conflict Graph
    # Conflict if dot > 2
    dots = V4 @ V4.T
    conflict_adj = [None] * N4
    for i in range(N4):
        if time.time() > deadline - 10:
            return best_points
        neighbors = np.where(dots[i] > 2)[0]
        conflict_adj[i] = neighbors
    del dots

    # 4. Structural Change: Bitset-accelerated Greedy with multiple restarts
    # We represent the "blocked" set as a list of Python integers (bitsets) or use numpy boolean arrays.
    # Given N4 ~ 5280, a numpy boolean array is fast.
    # We will perform many random greedy restarts.
    # To improve quality, we will also try a "weighted" greedy where weight is random, 
    # which is equivalent to random permutation but potentially faster if we can vectorize?
    # No, random permutation is already O(N).
    
    # Let's try a different structural approach: 
    # Use the property that the graph is highly symmetric?
    # Or just run the greedy much faster.
    
    # Optimization: Instead of `active[conflict_adj[v]] = False` which involves indexing a list of arrays,
    # we can pre-compute the neighbors as a list of lists for faster iteration if needed, 
    # but numpy boolean masking is quite fast.
    
    # Let's try to squeeze more out of the time by running more restarts and a simple local improvement.
    
    best_indices = []
    best_size = 0
    
    # Pre-compute neighbors as lists of lists for faster access in Python loops if we switch to pure Python bitsets?
    # Numpy boolean arrays are likely faster for N=5280.
    
    def greedy_with_order(order):
        active = np.ones(N4, dtype=bool)
        selected = []
        for v in order:
            if not active[v]:
                continue
            selected.append(v)
            active[v] = False
            if len(conflict_adj[v]) > 0:
                active[conflict_adj[v]] = False
        return selected

    # Run many random restarts
    # Estimate: Greedy takes ~ N4 * (avg_deg + 1) ops.
    # N4 = 5280. Avg degree?
    # Let's just run as many as possible.
    
    num_restarts = 0
    while time.time() < deadline - 20:
        order = np.random.permutation(N4)
        current = greedy_with_order(order)
        
        if len(current) > best_size:
            best_size = len(current)
            best_indices = current
            # If we found a new best, maybe do a quick local search?
            # For now, just track best.
        
        num_restarts += 1
        # If we hit a time limit, break
        if time.time() > deadline - 20:
            break

    # If we have time left, try to improve the best solution with local search
    if best_indices and time.time() < deadline - 10:
        # Simple local search on the best found
        in_set = np.zeros(N4, dtype=bool)
        for v in best_indices:
            in_set[v] = True
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding any vertex that has no conflicts with current set
            # This is a "1-add" move.
            # We can vectorize this:
            # For each vertex v not in set, check if any neighbor is in set.
            # This is equivalent to: for all v, if not in_set[v] and not (conflict_adj[v] intersects in_set)
            
            # Vectorized check:
            # Create a matrix M where M[i,j] = 1 if i and j conflict.
            # We already deleted `dots` but we have `conflict_adj`.
            # Reconstructing a full matrix is expensive.
            
            # Instead, iterate through candidates.
            candidates = np.where(~in_set)[0]
            if len(candidates) == 0:
                break
            
            # Shuffle to avoid bias
            np.random.shuffle(candidates)
            
            added = False
            for v in candidates:
                if in_set[v]:
                    continue
                # Check conflicts
                # If any neighbor is in set, skip
                if len(conflict_adj[v]) == 0:
                    in_set[v] = True
                    added = True
                    improved = True
                    continue
                
                # Check intersection
                # np.any(in_set[conflict_adj[v]])
                if not np.any(in_set[conflict_adj[v]]):
                    in_set[v] = True
                    added = True
                    improved = True
            
            if added:
                # Update best_indices?
                # We need to reconstruct list from in_set
                pass
            else:
                # No single addition possible. Try 1-swap?
                # Remove u, add v.
                # Gain if v has 0 conflicts in (set - u) and u has no other "critical" role?
                # Actually, if we remove u, we can add v if v has no conflicts in set \ {u}.
                # This means v's only conflict in the set was u.
                
                # Let's try: For each u in set, for each v in neighbors of u (that are not in set),
                # if v has no other neighbors in set, we can swap u for v.
                # Size remains same, but might allow further additions?
                # This is complex. Let's stick to simple additions for now.
                
                break
        
        # Reconstruct best_indices from in_set
        new_best_indices = np.where(in_set)[0]
        if len(new_best_indices) > best_size:
            best_size = len(new_best_indices)
            best_indices = new_best_indices

    if len(best_indices) > 0:
        new_points = V4[best_indices].astype(np.int64)
        best_points = new_points
        best_k = len(best_indices)

    return best_points
