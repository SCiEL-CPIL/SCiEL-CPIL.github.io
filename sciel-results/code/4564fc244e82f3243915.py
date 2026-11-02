import time
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110.0

    if k < 3:
        if k == 2:
            n = max(g, 3)
            return n, [(i, (i + 1) % n) for i in range(n)]
        return 1, []

    # Use Laszlo/LUW construction
    # q-regular bipartite graph on 2*q^d vertices with girth >= 2d+2
    q = k
    # Find d such that 2d + 2 >= g
    d = (g - 2 + 1) // 2
    if 2 * d + 2 < g:
        d += 1
        
    n = 2 * (q ** d)
    
    edges = []
    
    # Map integer index to vector (x1, ..., xd) in Z_q^d
    # We use mixed radix representation
    def index_to_vec(idx):
        vec = []
        for i in range(d):
            vec.append(idx % q)
            idx //= q
        return vec
    
    def vec_to_index(vec):
        idx = 0
        for i in range(d - 1, -1, -1):
            idx = idx * q + vec[i]
        return idx

    # Generate edges
    # Side 0: vertices 0 .. q^d - 1
    # Side 1: vertices q^d .. 2*q^d - 1
    
    for x_idx in range(q ** d):
        x = index_to_vec(x_idx)
        # For each y1 in Z_q
        for y1 in range(q):
            y = [0] * d
            y[0] = y1
            for i in range(1, d):
                # y_i = x_i + x_{i-1} * y_{i-1}
                y[i] = (x[i] + x[i-1] * y[i-1]) % q
            y_idx = vec_to_index(y)
            # Edge between x (side 0) and y (side 1)
            # Side 0 index is x_idx
            # Side 1 index is q^d + y_idx
            edges.append((x_idx, q ** d + y_idx))
            
    return n, edges

# EVOLVE-BLOCK-END
