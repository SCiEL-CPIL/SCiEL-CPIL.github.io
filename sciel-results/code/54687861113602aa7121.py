import numpy as np
import itertools
import time
from collections import deque

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    d = 11
    
    # Fallback: D11 minimal vectors (Norm 2) - 220 vectors
    # Type: two entries are +/-1, rest 0
    fallback_list = []
    for idxs in itertools.combinations(range(d), 2):
        i, j = idxs
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                fallback_list.append(v)
    fallback_points = np.array(fallback_list)
    
    # Try Norm 4 packing
    # Candidates: 
    # 1. One entry +/-2 (2*11 = 22 vectors)
    # 2. Four entries +/-1 (C(11,4) * 16 = 330 * 16 = 5280 vectors)
    # Total: 5302 candidates.
    
    # Actually, let's generate them.
    candidates_list = []
    
    # Type 1: +/-2
    for i in range(d):
        for s in [2, -2]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            candidates_list.append(v)
            
    # Type 2: +/-1 on 4 coords
    for idxs in itertools.combinations(range(d), 4):
        i, j, k, l = idxs
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    for sl in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        candidates_list.append(v)
                        
    candidates = np.array(candidates_list)
    n_cands = len(candidates)
    
    # Precompute squared norms
    # All have norm sq = 4.
    # Condition: min dist sq >= max norm sq = 4.
    # So we need pairwise dist sq >= 4.
    # Dist sq = ||u-v||^2 = ||u||^2 + ||v||^2 - 2u.v = 4 + 4 - 2u.v = 8 - 2u.v.
    # We need 8 - 2u.v >= 4 => 4 >= 2u.v => u.v <= 2.
    # So conflict if dot product > 2.
    
    # Build conflict graph using dot products.
    # N is 5302. N^2 is ~28M. 
    # We can do this in chunks or just use matrix multiplication.
    # Dot product matrix: D = candidates @ candidates.T (5302 x 5302)
    # This might be heavy memory wise (28M floats ~ 224MB). Acceptable.
    
    try:
        if time.time() < deadline - 60:
            dot_matrix = candidates @ candidates.T
            # Conflict if dot > 2
            conflict_mask = dot_matrix > 2
            # Diagonal is 4, so conflict_mask[i,i] is True.
            
            # We want Maximum Independent Set (MIS) on this graph.
            # Equivalent to Maximum Clique on complement graph.
            # Complement: compatible if dot <= 2.
            # compatible_mask = ~conflict_mask (but handle diagonal)
            # compatible_mask[i,j] = True if i != j and dot <= 2.
            
            compatible = dot_matrix <= 2
            # Set diagonal to False
            np.fill_diagonal(compatible, False)
            
            # Greedy coloring for upper bound / branch and bound?
            # Simple greedy:
            # 1. Randomize order
            # 2. Greedy add if compatible with all existing
            
            # Let's try a simple greedy with multiple restarts and random shuffling.
            # Also try a "degree-based" greedy: pick node with highest degree in complement (most compatible neighbors) first.
            
            best_size = 0
            best_indices = np.array([], dtype=np.int64)
            
            # Degree in compatible graph (number of compatible neighbors)
            degrees = compatible.sum(axis=1)
            
            # Strategy 1: Greedy by degree (desc)
            order = np.argsort(-degrees)
            current_set = []
            current_compatible = compatible[order].copy() # This is inefficient if we modify it, but let's see.
            
            # Better: maintain a set of candidates.
            # Since N is 5302, we can just do O(N^2) greedy.
            
            def greedy_greedy(order):
                selected = []
                # For each node in order, check if compatible with all selected
                # We can maintain a bitmask or just check.
                # With N=5302, max set size might be ~100-200?
                # Checking against 200 nodes for 5302 nodes is 1M ops. Fast.
                
                for idx in order:
                    if not current_time_check(deadline, start_time):
                        break
                    # Check compatibility with all selected
                    # selected contains indices into the original array.
                    # We need to check compatible[idx, sel] for all sel in selected.
                    # compatible is a bool matrix.
                    # compatible[idx, selected]
                    if len(selected) == 0:
                        selected.append(idx)
                    else:
                        # Check if compatible with all
                        # We can use np.all(compatible[idx, selected])
                        # This creates a view/array of size len(selected).
                        if np.all(compatible[idx, selected]):
                            selected.append(idx)
                return selected
            
            # Try greedy with degree order
            sel1 = greedy_greedy(order)
            best_size = max(best_size, len(sel1))
            if best_size == len(sel1):
                best_indices = np.array(sel1)
            
            # Try multiple random restarts with random order
            num_restarts = 50
            for r in range(num_restarts):
                if time.time() > deadline - 30:
                    break
                np.random.shuffle(order)
                # Maybe shuffle only a portion? No, full shuffle.
                sel_r = greedy_greedy(order)
                if len(sel_r) > best_size:
                    best_size = len(sel_r)
                    best_indices = np.array(sel_r)
                    # Try to improve by swapping?
                    # Simple local search: try to add any remaining node?
                    # Or try to swap a selected node with two unselected?
                    
            # If we found a set larger than fallback (220), return it.
            if best_size > 220:
                best_points = candidates[best_indices]
                # Validate
                norms_sq = np.sum(best_points**2, axis=1)
                max_norm = np.max(norms_sq)
                min_dist = float('inf')
                for i in range(len(best_points)):
                    for j in range(i+1, len(best_points)):
                        d_sq = np.sum((best_points[i] - best_points[j])**2)
                        if d_sq < min_dist:
                            min_dist = d_sq
                if min_dist >= max_norm:
                    return np.around(best_points).astype(np.int64)
    
    except Exception:
        pass
    
    return np.around(fallback_points).astype(np.int64)

def current_time_check(deadline, start_time):
    return time.time() < deadline - 5
