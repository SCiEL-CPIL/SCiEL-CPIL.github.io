import numpy as np
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    points_list = []
    
    # Generate all weight-4 vectors
    # Choose 4 indices out of 11
    for indices in combinations(range(d), 4):
        # Generate all 16 sign combinations
        for signs in product([1, -1], repeat=4):
            v = [0] * d
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            points_list.append(v)
            
    points = np.array(points_list, dtype=np.int64)
    points = np.around(points).astype(np.int64)
    
    return points
