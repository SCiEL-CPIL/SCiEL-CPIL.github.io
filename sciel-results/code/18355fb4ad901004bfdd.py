import numpy as np

D = 11

def construct():
    # Generate the D_11 lattice shell: all nonzero integer vectors in R^11
    # with exactly two nonzero entries, each +1 or -1.
    # This gives 4 * C(11,2) = 220 vectors of norm sqrt(2).
    # The minimum pairwise distance between distinct vectors is 2 (when they
    # share one coordinate with same sign and differ in the other, or are
    # orthogonal), which is >= sqrt(2) (the max norm). So the condition
    # "minimum pairwise distance >= max norm" holds.
    
    # Build all pairs (i, j) with i < j
    pairs = [(i, j) for i in range(D) for j in range(i+1, D)]
    
    # For each pair, create 4 sign combinations
    sign_combos = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    vectors = []
    for (i, j) in pairs:
        for (si, sj) in sign_combos:
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            v[j] = sj
            vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
