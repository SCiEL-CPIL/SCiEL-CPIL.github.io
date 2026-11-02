import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    D = 11
    
    # Generate Weight 2 and Weight 3 vectors
    # This reduces N from ~17600 to ~1540, allowing efficient matrix operations
    candidates = []
    
    # Weight 2
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            
    # Weight 3
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Compute dot products
    dots = np.dot(candidates, candidates.T)
    
    # Conflict condition: dist_sq < 4
    # For weight 2: norm_sq=4. dist_sq = 4+4-2*dots = 8-2*dots. Conflict if 8-2*dots < 4 => dots > 2.
    # For weight 3: norm_sq=9. dist_sq = 9+9-2*dots = 18-2*dots. Conflict if 18-2*dots < 9 => 2*dots > 9 => dots > 4.5 => dots >= 5.
    # Wait, the condition is "minimum pairwise distance is at least the maximum vector norm".
    # If we mix weights, max_norm is max(norm(v) for v in set).
    # If we pick only weight 2, max_norm = 2. dist >= 2 => dist_sq >= 4.
    # If we pick only weight 3, max_norm = 3. dist >= 3 => dist_sq >= 9.
    # If we mix, max_norm = 3 (assuming at least one weight 3). Then ALL pairs must have dist >= 3.
    # This is a much stricter constraint for weight 2 vectors if mixed with weight 3.
    # dist(v2, v3) >= 3 => dist_sq >= 9.
    # dist_sq(v2, v3) = 4 + 9 - 2*dots = 13 - 2*dots.
    # 13 - 2*dots >= 9 => 2*dots <= 4 => dots <= 2.
    
    # So if we mix, we need:
    # 1. Pair of W2: dots <= 2 (dist_sq >= 4, but max_norm is 3, so dist_sq >= 9 required? No.)
    # The condition is: for all i,j, ||v_i - v_j|| >= max_k ||v_k||.
    # Let M = max ||v_k||.
    # If we include W3, M=3.
    # Then ||v_i - v_j|| >= 3 for ALL i,j.
    # For two W2 vectors: ||v_i - v_j||^2 >= 9.
    # ||v_i - v_j||^2 = 8 - 2*dots.
    # 8 - 2*dots >= 9 => -2*dots >= 1 => dots <= -0.5 => dots <= -1.
    # For two W3 vectors: ||v_i - v_j||^2 >= 9.
    # ||v_i - v_j||^2 = 18 - 2*dots.
    # 18 - 2*dots >= 9 => 2*dots <= 9 => dots <= 4.5 => dots <= 4.
    # For W2 and W3: ||v_i - v_j||^2 >= 9.
    # ||v_i - v_j||^2 = 13 - 2*dots.
    # 13 - 2*dots >= 9 => 2*dots <= 4 => dots <= 2.
    
    # Strategy: Use only Weight 2 vectors to keep M=2.
    # Then condition is dist_sq >= 4.
    # This allows more vectors.
    # Let's stick to Weight 2 only for simplicity and density.
    # N = 220.
    
    # Re-generate candidates with Weight 2 only
    candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Compute dot products
    dots = np.dot(candidates, candidates.T)
    
    # Conflict if dist_sq < 4
    # dist_sq = 8 - 2*dots
    # 8 - 2*dots < 4 => 2*dots > 4 => dots > 2
    # Since dots are integers, dots >= 3 is conflict.
    # dots can be 2, 0, -2, etc.
    # If dots = 2, dist_sq = 4. Allowed.
    # If dots = 3, dist_sq = 2. Conflict.
    conflicts = dots >= 3
    np.fill_diagonal(conflicts, False)
    
    # Greedy Independent Set with Random Restarts and Local Search
    best_indices = []
    best_size = 0
    
    def greedy_independent_set(conflict_matrix, order):
        selected = []
        available = set(range(n_candidates))
        for v in order:
            if v in available:
                selected.append(v)
                # Remove v and its neighbors from available
                neighbors_to_remove = np.where(conflict_matrix[v])[0]
                for u in neighbors_to_remove:
                    if u in available:
                        available.remove(u)
                available.remove(v)
        return selected

    def local_search(selected, conflict_matrix):
        selected_set = set(selected)
        improved = True
        while improved:
            improved = False
            # Try adding any available node
            available = [i for i in range(n_candidates) if i not in selected_set and not np.any(conflict_matrix[i][list(selected_set)])]
            if available:
                # Add one with min degree in remaining
                min_deg = float('inf')
                best_v = -1
                for v in available:
                    deg = np.sum(conflict_matrix[v][available])
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                if best_v != -1:
                    selected.append(best_v)
                    selected_set.add(best_v)
                    improved = True
                    continue
                    
            # Try swaps
            # For each u in selected, remove it, then try to add 2 new nodes
            # This is expensive. Let's do 1-for-1 swaps first? No, size doesn't increase.
            # We need to increase size.
            # Try removing u and adding v1, v2 such that they don't conflict with new set.
            
            # Simpler: Random restarts are more effective for small N.
            # Let's just do a few random restarts.
            
        return selected

    # Run random restarts
    num_restarts = 1000
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        order = list(range(n_candidates))
        random.shuffle(order)
        
        # Greedy
        sel = greedy_independent_set(conflicts, order)
        
        # Simple Local Search: Try to add any valid node
        sel_set = set(sel)
        changed = True
        while changed:
            changed = False
            for v in range(n_candidates):
                if v not in sel_set:
                    # Check conflict with all in sel
                    if not np.any(conflicts[v][list(sel_set)]):
                        sel.append(v)
                        sel_set.add(v)
                        changed = True
                        if len(sel) > best_size:
                            best_size = len(sel)
                            best_indices = sel
        
        if len(sel) > best_size:
            best_size = len(sel)
            best_indices = sel

    if not best_indices:
        return np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    return candidates[best_indices]
