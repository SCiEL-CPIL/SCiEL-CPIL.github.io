import numpy as np
from itertools import combinations, product
import time

D = 11

def _generate_weight_k_vectors(k):
    """Generate all integer vectors in R^D with exactly k non-zero entries, each +/- 1."""
    vectors = []
    for pos in combinations(range(D), k):
        for signs in range(2**k):
            v = np.zeros(D, dtype=np.int64)
            for i, p in enumerate(pos):
                v[p] = 1 if (signs >> i) & 1 else -1
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def _greedy_independent_set(dots, max_dot, n_trials=50, strategy='random'):
    """
    Greedily select a subset of vectors such that pairwise dot product <= max_dot.
    dots: NxN matrix of dot products.
    """
    N = len(dots)
    if N == 0:
        return np.array([], dtype=np.int64)
    
    best_indices = []
    rng = np.random.default_rng(42)
    
    # Pre-compute conflict mask for faster lookup?
    # conflict_mask[i, j] = 1 if dots[i, j] > max_dot
    conflict_mask = dots > max_dot
    
    for trial in range(n_trials):
        current_indices = []
        current_set_mask = np.zeros(N, dtype=bool)
        
        if strategy == 'random':
            perm = rng.permutation(N)
        elif strategy == 'degree':
            # Calculate degree dynamically? No, static degree based on total conflicts
            # This is a heuristic
            degrees = conflict_mask.sum(axis=1)
            # Sort by degree, add random noise
            noise = rng.random(N) * 0.1
            order = np.argsort(degrees + noise)
            perm = order
        else:
            perm = rng.permutation(N)
            
        for idx in perm:
            if current_set_mask[idx]:
                continue
            # Check if idx conflicts with any in current set
            # current_set_mask has True for selected indices
            # We need to check conflict_mask[idx, current_set_mask]
            if np.any(conflict_mask[idx, current_set_mask]):
                continue
            current_indices.append(idx)
            current_set_mask[idx] = True
            
        if len(current_indices) > len(best_indices):
            best_indices = current_indices
            
    return np.array(best_indices, dtype=np.int64)

def construct():
    start_time = time.time()
    timeout = 115
    
    # Strategy: Try Weight 3 first, then Weight 4, then Mixed
    best_vectors = None
    best_k = 0
    
    # 1. Weight 3
    # Norm^2 = 3. Condition: dot <= 1.
    v3 = _generate_weight_k_vectors(3)
    dots3 = v3 @ v3.T
    idx3 = _greedy_independent_set(dots3, max_dot=1, n_trials=100, strategy='degree')
    if len(idx3) > best_k:
        best_vectors = v3[idx3]
        best_k = len(idx3)
        
    if time.time() - start_time > timeout:
        return best_vectors

    # 2. Weight 4
    # Norm^2 = 4. Condition: dot <= 2.
    v4 = _generate_weight_k_vectors(4)
    dots4 = v4 @ v4.T
    idx4 = _greedy_independent_set(dots4, max_dot=2, n_trials=100, strategy='degree')
    if len(idx4) > best_k:
        best_vectors = v4[idx4]
        best_k = len(idx4)

    if time.time() - start_time > timeout:
        return best_vectors

    # 3. Mixed Weight 3 and 4
    # This is complex. Let's stick to the best of pure sets for now, 
    # as mixed requires handling different max_dot rules per pair.
    # Given the time, pure sets are safer.
    
    return best_vectors
