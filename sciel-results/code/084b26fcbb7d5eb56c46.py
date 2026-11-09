import numpy as np
import time
import itertools
import random

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses randomized greedy search over a pool of low-weight integer vectors
    to find a large valid kissing configuration.
    """
    d = 11
    deadline = time.time() + 550  # 550s to leave margin before 600s kill
    
    # Generate candidate pool: vectors with 1, 2, or 3 non-zero entries (±1)
    candidates = []
    
    # Vectors with exactly 1 non-zero entry
    for i in range(d):
        for s in [-1, 1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    
    # Vectors with exactly 2 non-zero entries
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Vectors with exactly 3 non-zero entries
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        for sk in [-1, 1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
    
    n_candidates = len(candidates)
    candidates = np.array(candidates, dtype=np.int64)
    
    # Compute squared norms of all candidates
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    def is_valid_configuration(points):
        """Check if a set of points forms a valid kissing configuration."""
        if len(points) < 2:
            return False
        if any(np.all(p == 0) for p in points):
            return False
        
        # Maximum squared norm
        max_norm = max(np.sum(p ** 2) for p in points)
        
        # Minimum squared distance between any pair
        min_dist = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist_sq = np.sum(diff ** 2)
                if dist_sq < min_dist:
                    min_dist = dist_sq
        
        return min_dist >= max_norm
    
    def greedy_select(candidates, order):
        """Greedy selection: add points in given order if valid."""
        selected = []
        selected_norms = []
        
        for idx in order:
            if idx is None:
                continue
            v = candidates[idx]
            if np.all(v == 0):
                continue
            
            v_norm = int(np.sum(v ** 2))
            
            # Check against all selected points
            valid = True
            for sv in selected:
                diff = v - sv
                dist_sq = int(np.sum(diff ** 2))
                if dist_sq < v_norm or dist_sq < int(np.sum(sv ** 2)):
                    valid = False
                    break
            
            if valid:
                selected.append(v)
                selected_norms.append(v_norm)
        
        return np.array(selected, dtype=np.int64) if selected else np.empty((0, d), dtype=np.int64)
    
    best_result = np.empty((0, d), dtype=np.int64)
    best_size = 0
    
    # First, try deterministic greedy with norm-2 vectors only (fast baseline)
    norm2_indices = [i for i in range(n_candidates) if norms_sq[i] == 2]
    baseline = greedy_select(candidates, norm2_indices)
    if len(baseline) > best_size:
        best_result = baseline
        best_size = len(baseline)
    
    # Now try randomized greedy with multiple restarts
    random.seed(42)
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random shuffle of all candidates
        order = list(range(n_candidates))
        random.shuffle(order)
        
        result = greedy_select(candidates, order)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
    
    # If we still have time, try more targeted approaches
    if time.time() < deadline - 30:
        # Try greedy but prioritizing lower-norm vectors first
        for target_norm in [1, 2, 3, 4]:
            if time.time() > deadline - 10:
                break
            
            # Get candidates with this norm
            idxs = [i for i in range(n_candidates) if norms_sq[i] == target_norm]
            if not idxs:
                continue
            
            # Try several random orderings within this norm class
            for attempt in range(10):
                if time.time() > deadline - 10:
                    break
                random.shuffle(idxs)
                result = greedy_select(candidates, idxs)
                if len(result) > best_size:
                    best_size = len(result)
                    best_result = result
    
    if best_size == 0:
        # Fallback: return just norm-2 vectors
        return np.array([[1,1,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    return best_result
