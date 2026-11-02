"""Program for kissing number in dimension 11 using D_11 lattice minimal vectors."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools


def construct():
    # Generate minimal vectors of the D_11 lattice
    # These are vectors with exactly two non-zero entries, each +/-1
    base_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                base_vectors.append(v)
    
    vecs = np.array(base_vectors, dtype=np.int64)
    
    # The maximum norm squared is 2
    # The minimum pairwise distance squared is 2
    # Condition: min_dist_sq >= max_norm_sq => 2 >= 2 holds.
    
    return vecs
# EVOLVE-BLOCK-END
