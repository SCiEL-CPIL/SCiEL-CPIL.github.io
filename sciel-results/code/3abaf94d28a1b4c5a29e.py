import numpy as np
import itertools
import time

D = 11

def construct():
    # Generate the 220 shortest vectors of D11
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    
    vecs = np.array(vectors, dtype=np.int64)
    
    # Verification
    # Calculate norms
    norms = np.linalg.norm(vecs, axis=1)
    max_norm = np.max(norms)
    
    # Calculate pairwise distances
    # This is O(N^2 * D)
    # N=220, D=11. 220^2 * 11 is small.
    min_dist = np.inf
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            d = np.linalg.norm(vecs[i] - vecs[j])
            if d < min_dist:
                min_dist = d
                
    if min_dist >= max_norm:
        return vecs
    else:
        # Fallback: Norm 1 vectors
        fallback = []
        for i in range(D):
            for s in (1, -1):
                v = [0] * D
                v[i] = s
                fallback.append(v)
        return np.array(fallback, dtype=np.int64)
