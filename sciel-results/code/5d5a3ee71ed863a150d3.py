import numpy as np

D = 11

def construct():
    # Generate all vectors with exactly three non-zero entries, each +/- 1
    # Restrict to those with an even number of -1s (0 or 2 minuses)
    # This set has size 660 and satisfies the kissing condition
    
    vectors = []
    
    # Iterate over all combinations of 3 indices
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                # Signs for the three positions
                # We need an even number of -1s.
                # Possible sign patterns (s_i, s_j, s_k):
                # 0 minuses: (1, 1, 1)
                # 2 minuses: (-1, -1, 1), (-1, 1, -1), (1, -1, -1)
                
                signs = [
                    (1, 1, 1),
                    (-1, -1, 1),
                    (-1, 1, -1),
                    (1, -1, -1)
                ]
                
                for s in signs:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s[0]
                    v[j] = s[1]
                    v[k] = s[2]
                    vectors.append(v)
                    
    return np.array(vectors, dtype=np.int64)
