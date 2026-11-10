import numpy as np
import itertools
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill
    
    # Generate vectors with 4 non-zero entries (norm^2 = 4)
    vecs_4 = []
    for support in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for i, s in zip(support, signs):
                v[i] = s
            vecs_4.append(v)
    
    # Generate vectors with 6 non-zero entries (norm^2 = 6)
    vecs_6 = []
    for support in itertools.combinations(range(d), 6):
        for signs in itertools.product([1, -1], repeat=6):
            v = np.zeros(d, dtype=np.int64)
            for i, s in zip(support, signs):
                v[i] = s
            vecs_6.append(v)
    
    # Combine all vectors
    vectors = np.array(vecs_4 + vecs_6, dtype=np.int64)
    n4 = len(vecs_4)
    n6 = len(vecs_6)
    N_vecs = len(vectors)
    
    # Norm squared for each vector
    norm_sq = (vectors ** 2).sum(axis=1)
    
    # Conflict thresholds:
    # For pair (i, j): conflict if |v_i - v_j|^2 < max(|v_i|^2, |v_j|^2)
    # |v_i - v_j|^2 = |v_i|^2 + |v_j|^2 - 2*(v_i . v_j)
    # So: |v_i|^2 + |v_j|^2 - 2*(v_i . v_j) >= max(|v_i|^2, |v_j|^2)
    # => 2*(v_i . v_j) <= min(|v_i|^2, |v_j|^2)
    # => v_i . v_j <= min(|v_i|^2, |v_j|^2) / 2
    
    # For (4,4): threshold = 2
    # For (6,6): threshold = 3
    # For (4,6): threshold = 2
    
    # Pre-compute dot product matrix
    try:
        M = vectors @ vectors.T  # (N_vecs, N_vecs) int64 matrix
        
        # Build conflict matrix: conflict[i,j] = True if pair (i,j) violates constraint
        # We need: M[i,j] <= min(norm_sq[i], norm_sq[j]) / 2
        # Since all values are integers, we can use: 2*M[i,j] <= min(norm_sq[i], norm_sq[j])
        
        # Compute min_norm matrix
        min_norm = np.minimum(norm_sq[:, np.newaxis], norm_sq[np.newaxis, :])
        
        # Conflict if 2*M[i,j] > min_norm[i,j]
        conflict = (2 * M > min_norm)
        
        # Diagonal should be False (no self-conflict)
        np.fill_diagonal(conflict, False)
        
        best_size = 0
        best_indices = []
        
        # Multiple random restarts with greedy
        while time.time() < deadline:
            available = np.ones(N_vecs, dtype=bool)
            selected = []
            order = np.random.permutation(N_vecs)
            
            for idx in order:
                if available[idx]:
                    selected.append(idx)
                    # Remove all conflicting nodes from available
                    available &= ~conflict[idx, :]
                    
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_indices = selected[:]
            
            # Early stop if we've found a very good solution
            if best_size > 220:
                break
                
    except MemoryError:
        # Fallback: on-the-fly conflict checking
        best_size = 0
        best_indices = []
        
        while time.time() < deadline:
            available = np.ones(N_vecs, dtype=bool)
            selected = []
            order = np.random.permutation(N_vecs)
            
            for idx in order:
                if not available[idx]:
                    continue
                v = vectors[idx]
                ns = norm_sq[idx]
                ok = True
                for sel_idx in selected:
                    dot = int(np.dot(v, vectors[sel_idx]))
                    min_ns = min(ns, norm_sq[sel_idx])
                    if 2 * dot > min_ns:
                        ok = False
                        break
                if ok:
                    selected.append(idx)
                    # Mark conflicts for future (optional optimization)
                    for other_idx in range(N_vecs):
                        if not available[other_idx]:
                            continue
                        dot = int(np.dot(v, vectors[other_idx]))
                        min_ns = min(ns, norm_sq[other_idx])
                        if 2 * dot > min_ns:
                            available[other_idx] = False
                    
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_indices = selected[:]
            
            if best_size > 220:
                break

    # Construct the result
    if best_indices:
        points = vectors[best_indices]
    else:
        # Fallback to a minimal valid configuration
        points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
        
    return points
