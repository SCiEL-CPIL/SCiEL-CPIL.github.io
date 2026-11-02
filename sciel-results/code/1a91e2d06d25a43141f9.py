import numpy as np

D = 11

def construct():
    # Generate all integer vectors with exactly three non-zero entries, each +/- 1
    # Norm squared is 3.
    # We want a subset with min pairwise distance squared >= 3.
    # This implies u.dot(v) <= 1 for all u != v in the set.
    
    # Generate candidates
    candidates = []
    indices = np.arange(D)
    # Get all combinations of 3 indices
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                # 8 sign combinations
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        for s3 in [-1, 1]:
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            candidates.append(v)
    
    candidates = np.array(candidates)
    
    # Greedy selection
    selected = []
    for v in candidates:
        if not selected:
            selected.append(v)
            continue
        
        # Check dot product with all selected
        # v.dot(u) <= 1
        dots = selected @ v
        if np.all(dots <= 1):
            selected.append(v)
            
    if len(selected) > 220:
        return np.array(selected, dtype=np.int64)
    else:
        # Fallback to D11 roots (220 vectors)
        i_indices, j_indices = np.meshgrid(np.arange(D), np.arange(D), indexing='ij')
        mask = i_indices < j_indices
        i_list = i_indices[mask]
        j_list = j_indices[mask]
        signs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.int64)
        vectors = []
        for i, j in zip(i_list, j_list):
            for s in signs:
                v = np.zeros(D, dtype=np.int64)
                v[i] = s[0]
                v[j] = s[1]
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)
