"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Constructs a kissing configuration in R^11.
    Uses a set of 348 integer vectors with norm squared 8.
    The set consists of:
    1. All 220 vectors of the form (±2, ±2, 0, ..., 0).
    2. 128 vectors of the form (±1, ..., ±1, 0, 0, 0) supported on the first 8 dimensions,
       with an even number of -1s.
    
    This configuration satisfies min_dist >= max_norm (sqrt(8) >= sqrt(8)).
    """
    vectors = []
    
    # Part 1: All vectors with two ±2s and rest 0
    # There are C(11, 2) * 4 = 55 * 4 = 220 such vectors.
    for i in range(11):
        for j in range(i + 1, 11):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 11
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
                    
    # Part 2: Vectors with eight ±1s supported on indices 0-7, even parity of -1s
    # There are 2^7 = 128 such vectors.
    for mask in range(128):
        v = [0] * 11
        minus_count = 0
        for i in range(8):
            if (mask >> i) & 1:
                v[i] = -1
                minus_count += 1
            else:
                v[i] = 1
        # The condition "even number of -1s" is equivalent to the product of entries being 1?
        # No, E8 condition is even number of minuses.
        # My loop generates all 2^8 combinations.
        # I need to filter for even number of minuses.
        if minus_count % 2 == 0:
            vectors.append(v)

    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
