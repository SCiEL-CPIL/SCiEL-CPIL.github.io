# EVOLVE-BLOCK-START
import numpy as np
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    vecs = []
    # Use weight 4 vectors
    # Norm squared is 4.
    # Max dot product between distinct weight 4 vectors is 2.
    # Min dist squared = 4 + 4 - 2(2) = 4.
    # 4 >= 4 is valid.
    for positions in itertools.combinations(range(d), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            vecs.append(v)
            
    result = np.array(vecs, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
