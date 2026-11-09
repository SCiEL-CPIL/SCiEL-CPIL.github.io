import numpy as np
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # Generate all 4-subsets of indices
    all_4sets = list(combinations(range(d), 4))
    
    # Greedy selection of 4-sets such that no two share 3 or more indices
    # This ensures dot product <= 2 for any two vectors with different supports
    selected_supports = []
    for s in all_4sets:
        valid = True
        for s2 in selected_supports:
            # Check intersection size
            inter = len(set(s) & set(s2))
            if inter >= 3:
                valid = False
                break
        if valid:
            selected_supports.append(s)
            
    # Generate all sign combinations for each selected support
    for s in selected_supports:
        i, j, k, l = s
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        points.append(v)
                        
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
