"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    # Generate all weight-2 integer vectors in R^11
    # These vectors have two entries +/-1 and rest 0.
    # Norm squared is 2.
    # Condition for validity: min pairwise distance >= max norm.
    # Max norm is sqrt(2).
    # Distance squared between distinct weight-2 vectors u, v:
    # ||u-v||^2 = 2 + 2 - 2*u.v = 4 - 2*u.v
    # We need 4 - 2*u.v >= 2 => 2*u.v <= 2 => u.v <= 1.
    # For distinct weight-2 vectors, the maximum dot product is 1 
    # (when they share one index with the same sign).
    # If they share two indices, they are identical (dot=2) or negatives (dot=-2).
    # Identical vectors are not distinct.
    # So all distinct pairs satisfy u.v <= 1.
    # Thus, the set of ALL weight-2 vectors is a valid configuration.
    
    vectors = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
