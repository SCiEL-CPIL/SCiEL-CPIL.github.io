import numpy as np
import itertools
import time

def construct():
    D = 11
    # Generate all integer vectors with norm squared 3
    # These are vectors with exactly three non-zero entries, each being 1 or -1
    vectors = []
    indices = range(D)
    for i, j, k in itertools.combinations(indices, 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    v[k] = s3
                    vectors.append(v)
    
    # Shuffle to avoid bias
    np.random.shuffle(vectors)
    
    selected = []
    start_time = time.time()
    
    # Greedy selection
    # Condition: u . v <= 1 for all u, v in selected
    for v in vectors:
        if time.time() - start_time > 110: # Safety margin
            break
        compatible = True
        for u in selected:
            # Dot product
            # Since they are sparse, we can just use numpy dot
            if np.dot(u, v) > 1:
                compatible = False
                break
        if compatible:
            selected.append(v)
            
    if not selected:
        # Fallback to D11 roots if something went wrong or greedy failed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return np.array(selected, dtype=np.int64)
