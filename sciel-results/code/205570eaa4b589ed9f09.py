"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Generate all norm-2 vectors (two ±1 entries)
    norm2_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                norm2_vectors.append(v)
    
    selected = list(norm2_vectors)
    max_norm_sq = 4  # norm 2 squared
    
    # Try to add norm-sqrt(3) vectors (three ±1 entries)
    # For these, norm_sq = 3, but max_norm_sq will become 4 if we keep norm-2
    # Actually if we add norm-sqrt(3) vectors, max_norm is still 2 (since 2 > sqrt(3))
    # Wait: norm of norm-2 vector is 2, norm of norm-sqrt(3) vector is sqrt(3)≈1.73
    # So max_norm = 2, and we need distance >= 2 for all pairs
    
    # For a norm-2 vector u and norm-sqrt(3) vector v:
    # distance^2 = 4 + 3 - 2*u.v = 7 - 2*u.v
    # Need >= 4, so u.v <= 1.5, i.e., u.v <= 1
    
    # For two norm-sqrt(3) vectors u, v:
    # distance^2 = 3 + 3 - 2*u.v = 6 - 2*u.v
    # Need >= 4, so u.v <= 1
    
    # Generate norm-sqrt(3) vectors
    norm3_vectors = []
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j], v[k] = si, sj, sk
                    norm3_vectors.append(v)
    
    # Greedily add norm-3 vectors that satisfy constraints
    selected_arr = np.array(selected)
    
    for v in norm3_vectors:
        if time.time() > deadline:
            break
        v = v.astype(np.int64)
        # Check against all selected vectors
        # For norm-2 vectors: u.v <= 1
        # For norm-3 vectors: u.v <= 1
        dots = selected_arr @ v
        if np.all(dots <= 1):
            selected.append(v)
            selected_arr = np.array(selected)
    
    return np.array(selected, dtype=np.int64)
# EVOLVE-BLOCK-END
