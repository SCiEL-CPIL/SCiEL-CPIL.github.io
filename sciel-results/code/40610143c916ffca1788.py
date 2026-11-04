import time
import random
import math
from collections import deque

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s margin

    # LUW parameters
    q = k
    
    if g == 10:
        d = 4
    elif g == 8:
        d = 3
    elif g == 6:
        d = 2
    else:
        d = (g - 2) // 2 + 1

    n = 2 * (q ** d)
    
    # GF(q) arithmetic
    if q == 4:
        # GF(4) tables
        add_table = [
            [0,1,2,3],
            [1,0,3,2],
            [2,3,0,1],
            [3,2,1,0]
        ]
        mul_table = [
            [0,0,0,0],
            [0,1,2,3],
            [0,2,3,1],
            [0,3,1,2]
        ]
        def gf_add(x, y): return add_table[x][y]
        def gf_mul(x, y): return mul_table[x][y]
    elif q == 2:
        def gf_add(x, y): return (x + y) % 2
        def gf_mul(x, y): return (x * y) % 2
    else:
        # Assume q is prime
        def gf_add(x, y): return (x + y) % q
        def gf_mul(x, y): return (x * y) % q

    # Vertex indexing:
    # (0, x) -> index = encode(x)
    # (1, x) -> index = q^d + encode(x)
    
    def encode(x):
        idx = 0
        for i in range(d):
            idx = idx * q + x[i]
        return idx

    def decode(idx):
        x = [0] * d
        for i in range(d-1, -1, -1):
            x[i] = idx % q
            idx //= q
        return x

    edges = set()
    
    # Generate edges from side 0 to side 1
    q_d = q ** d
    for x_idx in range(q_d):
        x = decode(x_idx)
        v0 = x_idx
        
        for a in range(q):
            # Calculate y from x and a
            y = [0] * d
            y[0] = a
            for i in range(1, d):
                # y_i = x_i + a * y_{i-1}
                y[i] = gf_add(x[i], gf_mul(a, y[i-1]))
            
            y_idx = encode(y)
            v1 = q_d + y_idx
            
            if v0 < v1:
                edges.add((v0, v1))
            else:
                edges.add((v1, v0))
                
    edge_list = list(edges)
    
    return n, edge_list
