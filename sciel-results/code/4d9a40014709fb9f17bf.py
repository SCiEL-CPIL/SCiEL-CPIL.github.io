"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    """
    Construct a kissing configuration in R^11 using the roots of the D_11 lattice.
    
    The roots of D_n are vectors with exactly two non-zero entries, each being +1 or -1.
    For n=11, there are C(11,2) * 4 = 55 * 4 = 220 such vectors.
    
    All these vectors have norm sqrt(2).
    The minimum pairwise distance between any two distinct roots is sqrt(2).
    (Proof: ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v> = 4 - 2<u,v>. 
     Max <u,v> for distinct roots is 1, so min ||u-v||^2 = 2, min ||u-v|| = sqrt(2).)
    
    The maximum vector norm is sqrt(2).
    Condition: min pairwise distance >= max vector norm 
    => sqrt(2) >= sqrt(2), which holds.
    
    This yields a valid configuration of size 220.
    """
    vectors = []
    
    # Generate all pairs of indices
    for i, j in itertools.combinations(range(D), 2):
        # Generate all 4 sign combinations for the two non-zero entries
        for s_i in [1, -1]:
            for s_j in [1, -1]:
                v = [0] * D
                v[i] = s_i
                v[j] = s_j
                vectors.append(v)
    
    # Convert to numpy array
    result = np.array(vectors, dtype=np.int64)
    
    return result
# EVOLVE-BLOCK-END
