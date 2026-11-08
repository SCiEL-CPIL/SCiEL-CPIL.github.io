"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    vectors = []
    
    # Baseline: all vectors with exactly two non-zero entries of ±1
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    
    vecs = np.array(vectors, dtype=np.int64)
    n = len(vecs)
    
    # Precompute squared norms
    norms_sq = np.sum(vecs * vecs, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Greedy addition: try vectors with small integer entries
    # We'll try vectors with entries in {-2, -1, 0, 1, 2} but limit search
    # by checking if the vector can be added (min distance >= max norm)
    
    # To make this tractable, we'll try structured families:
    # 1. Vectors with one non-zero entry of magnitude 2
    # 2. Vectors with three non-zero entries of ±1
    # 3. Vectors with two non-zero entries where one is ±2
    
    candidates = []
    
    # Family 1: single entry ±2
    for i in range(D):
        for s in (2, -2):
            v = [0] * D
            v[i] = s
            candidates.append(v)
    
    # Family 2: three non-zero entries of ±1
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
    
    # Family 3: two non-zero entries, one is ±2, other is ±1
    for i, j in itertools.combinations(range(D), 2):
        for s2 in (2, -2):
            for s1 in (1, -1):
                v = [0] * D
                v[i], v[j] = s2, s1
                candidates.append(v)
                v2 = [0] * D
                v2[i], v2[j] = s1, s2
                candidates.append(v2)
    
    # Try to add each candidate if it satisfies the distance constraint
    for cand in candidates:
        if time.time() > deadline - 5:
            break
        
        cv = np.array(cand, dtype=np.int64)
        cv_norm_sq = np.sum(cv * cv)
        
        # Check: min pairwise distance >= max norm
        # For new vector cv, we need:
        # 1. dist(cv, v_i) >= max(norm(cv), norm(v_i)) for all existing v_i
        # 2. After adding, the new max norm might increase
        
        # dist(cv, v_i)^2 = norm(cv)^2 + norm(v_i)^2 - 2*dot(cv, v_i)
        # We need dist(cv, v_i) >= max(norm(cv), norm(v_i))
        # i.e., dist(cv, v_i)^2 >= max(norm(cv)^2, norm(v_i)^2)
        
        dots = vecs @ cv
        dists_sq = cv_norm_sq + norms_sq - 2 * dots
        
        # Condition: dists_sq >= max(cv_norm_sq, norms_sq) element-wise
        # max(cv_norm_sq, norms_sq) is element-wise max
        min_required = np.maximum(cv_norm_sq, norms_sq)
        
        if np.all(dists_sq >= min_required):
            vecs = np.vstack([vecs, cv])
            norms_sq = np.append(norms_sq, cv_norm_sq)
            max_norm_sq = np.max(norms_sq)
    
    return vecs
# EVOLVE-BLOCK-END
