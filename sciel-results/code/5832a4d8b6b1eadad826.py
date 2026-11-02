import numpy as np
import itertools
import time

D = 11

def generate_vectors(w):
    vectors = []
    # Generate all combinations of indices
    for indices in itertools.combinations(range(D), w):
        # Generate all sign combinations
        for signs in itertools.product([1, -1], repeat=w):
            v = [0] * D
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            vectors.append(v)
    return np.array(vectors, dtype=np.int8)

def find_independent_set(vectors, w):
    N = len(vectors)
    # Compute dot products
    # vectors shape (N, D)
    dots = vectors @ vectors.T
    threshold = w / 2.0
    
    # Conflict matrix: True if conflict
    # Conflict if dot > threshold
    # Since dots are integers, dot > w/2
    if w % 2 == 0:
        # threshold is integer, e.g. 2.0
        # Conflict if dot >= threshold + 1 ?
        # Condition: dot <= threshold.
        # So conflict if dot > threshold.
        # If threshold=2.0, conflict if dot >= 3.
        conflict_mask = dots > threshold
    else:
        # threshold is x.5, e.g. 1.5
        # Conflict if dot > 1.5 => dot >= 2.
        conflict_mask = dots > threshold
        
    # Diagonal is always conflict (self), but we ignore it
    np.fill_diagonal(conflict_mask, False)
    
    # Greedy Min-Degree
    # degrees: number of conflicts for each vertex
    degrees = conflict_mask.sum(axis=1)
    
    selected = []
    available = np.ones(N, dtype=bool)
    
    # Optimization: use a heap or just argmin
    # Since N is ~5000, argmin is fast enough
    
    while available.sum() > 0:
        # Find available vertex with min degree
        # Set degrees of unavailable to inf
        current_degrees = np.where(available, degrees, np.inf)
        idx = np.argmin(current_degrees)
        
        if not available[idx]:
            break
            
        selected.append(idx)
        
        # Remove idx and its neighbors
        # Neighbors are those where conflict_mask[idx, :] is True
        neighbors = conflict_mask[idx]
        
        # Update available
        available[idx] = False
        available[neighbors] = False
        
        # Update degrees
        # For all remaining available vertices, subtract 1 if they were connected to idx
        # Actually, we just need to update the degrees of the neighbors of idx
        # But since we are removing neighbors, their degree doesn't matter for future selection
        # We only care about degrees of vertices that remain available
        # The degree of a remaining vertex v decreases by 1 if it was connected to idx
        # So degrees[available] -= conflict_mask[idx, available]
        
        # This is a vectorized update
        degrees[available] -= conflict_mask[idx, available]
        
    return selected

def construct():
    start_time = time.time()
    deadline = 100.0 # Safety margin
    
    best_set = []
    best_w = 0
    
    # Try weights 3, 4, 5
    for w in [3, 4, 5]:
        if time.time() > deadline - 10:
            break
            
        vectors = generate_vectors(w)
        N = len(vectors)
        
        # If N is too large for memory, skip or use block method
        # 30000x30000 is 3.6GB, too big.
        # 5000x5000 is 100MB, ok.
        # 15000x15000 is 900MB, risky.
        if N > 20000:
            continue
            
        selected = find_independent_set(vectors, w)
        
        if len(selected) > len(best_set):
            best_set = selected
            best_w = w
            
    # If no set found, fall back to seed
    if not best_set:
        # Seed logic
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    # Construct final array
    # We need the actual vectors
    # We generated vectors for w=3,4,5 in the loop, but didn't store them all.
    # We need to regenerate or store.
    # Let's regenerate for best_w
    vectors = generate_vectors(best_w)
    result = vectors[best_set]
    
    return result
