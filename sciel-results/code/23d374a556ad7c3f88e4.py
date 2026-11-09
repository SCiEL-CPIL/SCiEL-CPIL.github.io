import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Fallback: The original solution
    vectors_orig = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors_orig.append(v)
    orig_set = np.array(vectors_orig, dtype=np.int64)
    k_orig = len(orig_set)
    
    best_vectors = orig_set
    best_k = k_orig
    
    # Try Weight 3 vectors
    # Generate all vectors with exactly 3 non-zero entries (+/-1)
    # Number of such vectors: C(11,3) * 2^3 = 165 * 8 = 1320
    
    vecs = []
    indices = []
    for combo in itertools.combinations(range(D), 3):
        i, j, k_idx = combo
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k_idx] = s1, s2, s3
                    vecs.append(v)
                    indices.append((i, j, k_idx, s1, s2, s3))
                    
    V = np.array(vecs, dtype=np.int64)
    N = len(V)
    
    if N == 0:
        return best_vectors
        
    # Calculate pairwise dot products
    # V is (N, D). Dot product matrix is V @ V.T
    # We need to find pairs (i, j) with i != j such that V[i] @ V[j] >= 2
    # Since V[i] @ V[i] = 3, we can compute the full matrix.
    
    dots = V @ V.T
    
    # Conflict if dots[i, j] >= 2 and i != j
    # Create adjacency list or matrix
    # N is 1320, so N x N matrix is 1.7M entries. Fast enough.
    
    # Mask out diagonal
    np.fill_diagonal(dots, -1) # Ensure diagonal is not a conflict
    
    conflicts = dots >= 2
    
    # Greedy Independent Set
    # 1. Compute degrees
    degrees = np.sum(conflicts, axis=1)
    
    # 2. Order vertices by degree (ascending)
    order = np.argsort(degrees)
    
    selected = np.zeros(N, dtype=bool)
    removed = np.zeros(N, dtype=bool)
    
    for v_idx in order:
        if time.time() > deadline - 5.0:
            break
        if removed[v_idx]:
            continue
        selected[v_idx] = True
        # Remove neighbors
        neighbors = np.where(conflicts[v_idx])[0]
        removed[neighbors] = True
        removed[v_idx] = True
        
    k_new = np.sum(selected)
    
    if k_new > best_k:
        best_vectors = V[selected]
        best_k = k_new
        
    return best_vectors
