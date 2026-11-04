import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    
    # Generate supports
    supports = list(itertools.combinations(range(D), 4))
    n = len(supports)
    
    # Bitmasks
    masks = []
    for sup in supports:
        m = 0
        for idx in sup:
            m |= (1 << idx)
        masks.append(m)
        
    # Compatibility matrix
    # adj[i] is a set of neighbors
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if (masks[i] & masks[j]).bit_count() <= 2:
                adj[i].add(j)
                adj[j].add(i)
                
    # Max Clique using Tomita's algorithm
    # ...
    
    # If timeout, return best found
    # ...
    
    # Construct vectors
    vectors = []
    for sup in clique:
        # Get indices
        idxs = supports[sup]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in zip(idxs, signs):
                v[k] = s
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
