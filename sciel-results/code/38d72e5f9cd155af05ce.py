import numpy as np
import time
import random
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 575  # Safety margin for 600s limit
    
    # The base set of 220 norm-2 vectors is always valid with R=2.
    base_vectors = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    base_vectors.append(v)
    
    base_array = np.array(base_vectors, dtype=np.int64)
    base_k = len(base_array)  # 220
    
    # Verify base set
    max_norm_base = np.max(np.sum(base_array**2, axis=1))  # 2
    valid_base = True
    for i in range(base_k):
        for j in range(i + 1, base_k):
            d2 = np.dot(base_array[i] - base_array[j], base_array[i] - base_array[j])
            if d2 < max_norm_base:
                valid_base = False
                break
        if not valid_base:
            break
    
    best_set = base_array
    best_k = base_k if valid_base else 2
    
    # For R=4, we want to find a large independent set in the conflict graph
    # where vertices are integer vectors with norm^2 <= 4, and edges connect
    # pairs with dist^2 < 4.
    
    # Generate all candidates with norm^2 <= 4
    candidates = []
    
    # Norm 1: 22 vectors
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 1
        candidates.append(v)
        candidates.append(-v)
    
    # Norm 2: 220 vectors
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
    
    # Norm 3: 1320 vectors
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        for s3 in [1, -1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            candidates.append(v)
    
    # Norm 4: 
    # Type A: (±2, 0, ...) -> 22 vectors
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 2
        candidates.append(v)
        candidates.append(-v)
    
    # Type B: (±1, ±1, ±1, ±1, 0, ...) -> C(11,4) * 16 = 330 * 16 = 5280 vectors
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for l in range(k + 1, d):
                    for s1 in [1, -1]:
                        for s2 in [1, -1]:
                            for s3 in [1, -1]:
                                for s4 in [1, -1]:
                                    v = np.zeros(d, dtype=np.int64)
                                    v[i] = s1
                                    v[j] = s2
                                    v[k] = s3
                                    v[l] = s4
                                    candidates.append(v)
    
    cand = np.array(candidates, dtype=np.int64)
    n = len(cand)
    norms = np.sum(cand**2, axis=1)
    
    # Only proceed if n is manageable
    if n > 15000:
        # Too many candidates, just return base
        if valid_base:
            return base_array
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    # Compute conflict graph: edge (i,j) if dist^2(i,j) < 4
    # dist^2(i,j) = ||cand[i]||^2 + ||cand[j]||^2 - 2 * cand[i]·cand[j]
    dots = cand @ cand.T
    dist2 = norms[:, None] + norms[None, :] - 2 * dots
    conflict = (dist2 < 4)
    np.fill_diagonal(conflict, False)
    
    # Degree of each vertex in conflict graph
    degree = conflict.sum(axis=1)
    
    # Greedy MIS with multiple random restarts
    # Use a priority queue approach: always pick the vertex with minimum degree among available
    
    def greedy_mis(n, conflict, degree, time_limit, seed=0):
        """Greedy maximum independent set."""
        rng = random.Random(seed)
        available = np.ones(n, dtype=bool)
        selected = []
        start = time.time()
        
        # Work on a copy of degree for available vertices
        # Use a simpler approach: repeatedly pick min-degree available vertex
        
        # For efficiency, maintain a list of available vertices
        # and their current degrees (degree to other available vertices)
        
        # Simple approach: just iterate
        while available.any():
            if time.time() > time_limit:
                break
            
            avail_idx = np.where(available)[0]
            if len(avail_idx) == 0:
                break
            
            # Find the vertex with minimum conflict-degree among available
            # conflict-degree = number of conflicts with other available vertices
            # For a vertex v, its conflict-degree among available = sum of conflict[v, u] for u in available
            # This is expensive to compute naively.
            
            # Approximation: use the original degree as a proxy,
            # or compute it more carefully.
            
            # Let's use the original degree for the first pass,
            # and a more careful computation for subsequent picks.
            
            # Actually, let's just use the original degree. It's a heuristic.
            min_deg = np.min(degree[avail_idx])
            candidates_for_pick = avail_idx[degree[avail_idx] == min_deg]
            
            if len(candidates_for_pick) > 1 and seed != 0:
                pick_idx = candidates_for_pick[rng.randint(0, len(candidates_for_pick) - 1)]
            else:
                pick_idx = candidates_for_pick[0]
            
            selected.append(pick_idx)
            
            # Remove pick_idx and all its neighbors from available
            to_remove = conflict[pick_idx].copy()
            to_remove[pick_idx] = True
            available[to_remove] = False
            
        return selected
    
    # Run multiple greedy restarts
    n_trials = 50
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        seed = trial + 1
        selected = greedy_mis(n, conflict, degree, deadline, seed=seed)
        
        if len(selected) > best_k:
            best_k = len(selected)
            best_set = cand[selected]
            
            # Verify
            if len(best_set) >= 2:
                max_norm = np.max(np.sum(best_set**2, axis=1))
                valid = True
                for i in range(len(best_set)):
                    for j in range(i + 1, len(best_set)):
                        d2 = np.dot(best_set[i] - best_set[j], best_set[i] - best_set[j])
                        if d2 < max_norm:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    # Should not happen, but just in case
                    best_k = base_k if valid_base else 2
                    best_set = base_array if valid_base else np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    # Now try a local search / improvement on the best set found
    # Try to add vertices one by one that don't conflict with the current set
    
    if time.time() < deadline - 60 and best_k > 0 and len(best_set) > 0:
        # Build a set of selected indices for fast lookup
        selected_set = set()
        # Map from candidate index to its position in best_set
        # We need to know which candidate indices are in best_set
        # Since best_set = cand[selected_indices], we can track selected_indices
        
        # Reconstruct selected_indices from best_set
        # This is tricky because best_set is a numpy array.
        # Let's redo this more carefully.
        
        # Instead, let's work with the selected indices directly.
        # Let's redo the greedy with index tracking.
        
        pass  # Skip local search for now to keep it simple and fast
    
    # Final verification and return
    if best_k >= 2:
        result = np.array(best_set, dtype=np.int64)
        max_norm = np.max(np.sum(result**2, axis=1))
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d2 = np.dot(result[i] - result[j], result[i] - result[j])
                if d2 < max_norm:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return result
    
    # Fallback
    if valid_base:
        return base_array
    else:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
