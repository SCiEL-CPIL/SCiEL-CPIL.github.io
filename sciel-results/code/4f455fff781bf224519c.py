import numpy as np
import itertools
import time
import random

def generate_norm2_vectors(d=11):
    """
    Generate all integer vectors in Z^d with squared norm 2.
    These are vectors with exactly two non-zero entries, each +/- 1.
    """
    vecs = []
    for idxs in itertools.combinations(range(d), 2):
        i, j = idxs
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                vecs.append(v)
    return np.array(vecs)

def generate_norm4_vectors(d=11):
    """
    Generate all integer vectors in Z^d with squared norm 4.
    Types:
    1. (2, 0, ...) : 22 vectors
    2. (1, 1, 1, 1, ...) : 5280 vectors
    """
    vecs = []
    # Type 1: One coordinate is +/- 2
    for i in range(d):
        for s in [1, -1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = 2 * s
            vecs.append(v)
    
    # Type 2: Four coordinates are +/- 1
    for idxs in itertools.combinations(range(d), 4):
        i, j, k, l = idxs
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    for s4 in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        v[k] = s3
                        v[l] = s4
                        vecs.append(v)
    return np.array(vecs)

def validate_configuration(points):
    """
    Validate the configuration according to the task contract.
    """
    if points.shape[1] != 11:
        return False
    k = points.shape[0]
    if k < 2:
        return False
    
    # Check no zero vector
    if np.any(np.all(points == 0, axis=1)):
        return False
        
    # Calculate max squared norm
    norms_sq = np.sum(points**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Calculate min squared distance
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            d_sq = np.sum(diff**2)
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                
    # Condition: min_dist_sq >= max_norm_sq
    return min_dist_sq >= max_norm_sq

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580 # Safety margin
    
    # 1. Generate Candidate Pool
    # We consider vectors with norm 2 and norm 4.
    # If we select a set S, let R = max_{v in S} ||v||^2.
    # We require min_{u,v in S, u!=v} ||u-v||^2 >= R.
    
    # Strategy:
    # Since mixing norms forces R to be the max norm present, 
    # and norm 4 vectors are "far" from each other potentially, 
    # we try to find a large set where R=4.
    # This requires all pairwise distances >= 4.
    
    # Pool: Norm 2 and Norm 4 vectors.
    norm2_vecs = generate_norm2_vectors(11)
    norm4_vecs = generate_norm4_vectors(11)
    
    # Combine pool
    # We will represent the problem as finding a Max Independent Set
    # in the graph G where nodes are candidates and edges exist if dist^2 < 4.
    # Why 4? Because if we include any norm 4 vector, R=4.
    # If we include ONLY norm 2 vectors, R=2, and we need dist >= 2.
    # The max independent set in G_dist<2 for norm 2 vectors is the whole set (220).
    # But we want to beat 220.
    # So we must include norm 4 vectors, which forces R=4.
    # Thus, we solve for Max Independent Set in G_dist<4 over the combined pool.
    
    pool = np.vstack([norm2_vecs, norm4_vecs])
    N = len(pool)
    
    # Precompute squared distances? N is ~5500. N^2 is 30M.
    # 30M floats is 240MB. Manageable.
    # But we can do it smarter.
    
    # We will use a Greedy approach with a "conflict" check.
    # To make it fast, we precompute the dot products or use vectorized distance checks.
    
    best_k = 220
    best_points = norm2_vecs.copy()
    
    # Multiple random restarts
    # Shuffle the pool to break symmetries and find different local optima
    
    max_restarts = 200
    restart_count = 0
    
    # Optimization: 
    # Instead of checking all pairs, we maintain a list of selected points.
    # For each candidate, we check if it conflicts with any selected point.
    # Conflict if dist^2 < 4.
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle indices
        indices = np.arange(N)
        np.random.shuffle(indices)
        
        selected_indices = []
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            cand = pool[idx]
            
            # Check against all selected
            # We can vectorize this if we keep selected in an array
            # But selected grows.
            # For speed, we keep a list of selected vectors.
            
            is_valid = True
            for sel_idx in selected_indices:
                sel_vec = pool[sel_idx]
                diff = cand - sel_vec
                dist_sq = np.dot(diff, diff)
                if dist_sq < 4:
                    is_valid = False
                    break
                    
            if is_valid:
                selected_indices.append(idx)
                
        k_sel = len(selected_indices)
        
        if k_sel > best_k:
            best_k = k_sel
            best_points = pool[selected_indices].copy()
            
        restart_count += 1
        
    # Final Validation
    if best_points.shape[0] > 0:
        points = np.around(best_points).astype(np.int64)
        if validate_configuration(points):
            return points
            
    # Fallback to Norm 2
    points = np.around(norm2_vecs).astype(np.int64)
    if validate_configuration(points):
        return points
        
    # Absolute fallback
    fallback = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int64)
    return fallback
