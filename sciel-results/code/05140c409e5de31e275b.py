"""Construction of a kissing configuration in R^11 using integer vectors."""
import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # safety margin from 150s limit

    # Strategy: Generate a diverse candidate pool of integer vectors with
    # varying weights, then use a greedy independent set algorithm to select
    # the maximum number of vectors satisfying the distance constraint.
    #
    # Constraint: min_{i != j} ||v_i - v_j|| >= max_i ||v_i||
    # In squared terms: min dist^2 >= max norm^2
    #
    # For vectors with the same norm^2 = N, we need:
    #   ||v_i - v_j||^2 = 2N - 2*v_i·v_j >= N
    #   => v_i·v_j <= N/2
    #
    # We'll build candidates in order of increasing norm^2, and for each
    # candidate, check if it's compatible with all already-selected vectors.
    
    # Generate candidate vectors by weight
    candidates = []
    
    # Weight 1: norm^2 = 1, dot product threshold: <= 0.5, so dot <= 0
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append((1, v))  # (norm_sq, vector)
    
    # Weight 2: norm^2 = 2, dot product threshold: <= 1, so dot <= 1
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append((2, v))
    
    # Weight 3: norm^2 = 3, dot product threshold: <= 1.5, so dot <= 1
    for i, j, k in combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append((3, v))
    
    # Sort by norm squared (lower first) to maximize count
    candidates.sort(key=lambda x: x[0])
    
    # Greedy independent set: select vectors that are compatible with all
    # previously selected ones
    selected = []
    selected_norms_sq = []
    
    for norm_sq, v in candidates:
        if time.time() > deadline:
            break
        
        # Check compatibility with all selected vectors
        compatible = True
        for sel_norm_sq, sel_v in selected:
            # Compute dot product
            dot = np.dot(v, sel_v)
            # For two vectors with norms a and b, distance^2 = a^2 + b^2 - 2*dot
            # We need: a^2 + b^2 - 2*dot >= max(a^2, b^2)
            # WLOG assume a^2 >= b^2, then need: a^2 + b^2 - 2*dot >= a^2
            # => b^2 - 2*dot >= 0
            # => dot <= b^2 / 2
            
            max_norm_sq = max(norm_sq, sel_norm_sq)
            min_norm_sq = min(norm_sq, sel_norm_sq)
            
            # Need: norm_sq + sel_norm_sq - 2*dot >= max_norm_sq
            # => min_norm_sq - 2*dot >= 0
            # => dot <= min_norm_sq / 2
            
            if dot > min_norm_sq / 2.0:
                compatible = False
                break
        
        if compatible:
            selected.append((norm_sq, v))
    
    if time.time() > deadline:
        # Return what we have
        if selected:
            result = np.array([v for _, v in selected], dtype=np.int64)
            return result
        else:
            # Fallback to weight-2 only
            vectors = []
            for i in range(D):
                for j in range(i + 1, D):
                    for si in (1, -1):
                        for sj in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            vectors.append(v)
            return np.array(vectors, dtype=np.int64)
    
    if selected:
        result = np.array([v for _, v in selected], dtype=np.int64)
        return result
    else:
        # Fallback to weight-2 only
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        vectors.append(v)
        return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
