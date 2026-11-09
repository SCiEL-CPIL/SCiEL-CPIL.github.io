import numpy as np
import time
from itertools import combinations, product
import random

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580  # safety margin before 630s limit
    
    # Known 110-point solution: all vectors with exactly 2 non-zero entries (±1)
    # norm^2 = 2, min dist^2 = 2. Valid.
    
    # Try to find a better solution using vectors with exactly 3 non-zero entries (±1)
    # norm^2 = 3, need pairwise dist^2 >= 3, i.e., dot <= 1
    
    # Generate all candidates with exactly 3 non-zero entries
    candidates = []
    for pos in combinations(range(d), 3):
        for signs in product([-1, 1], repeat=3):
            v = np.zeros(d, dtype=np.int64)
            for p, s in zip(pos, signs):
                v[p] = s
            candidates.append(v)
    
    n_cands = len(candidates)  # 1320
    
    # Precompute all-pairs dot products
    # dot_matrix[i][j] = candidates[i] . candidates[j]
    dot_matrix = np.zeros((n_cands, n_cands), dtype=np.int64)
    for i in range(n_cands):
        for j in range(i+1, n_cands):
            dot = np.dot(candidates[i], candidates[j])
            dot_matrix[i, j] = dot
            dot_matrix[j, i] = dot
    
    # Greedy: try multiple random orderings, pick best
    random.seed(42)
    best_k = 0
    best_selected = None
    
    n_trials = 200
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        order = list(range(n_cands))
        random.shuffle(order)
        
        selected = []
        selected_set = set()
        
        for idx in order:
            # Check if candidate idx is compatible with all selected
            compatible = True
            for s_idx in selected:
                if dot_matrix[idx, s_idx] > 1:
                    compatible = False
                    break
            if compatible:
                selected.append(idx)
                selected_set.add(idx)
        
        k = len(selected)
        if k > best_k:
            best_k = k
            best_selected = selected
    
    # Build the best solution
    if best_selected is not None and best_k > 110:
        points = np.array([candidates[i] for i in best_selected], dtype=np.int64)
        
        # Verify
        k = points.shape[0]
        norms = np.sum(points**2, axis=1)
        max_norm = np.max(norms)
        
        valid = True
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
    
    # Fallback: 110-point solution
    candidates_2 = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates_2.append(v)
    
    return np.array(candidates_2, dtype=np.int64)
