import numpy as np
import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 140s limit
    
    D = 11
    vectors = []
    
    # 1. Generate E8 integer vectors of norm squared 4 in first 8 dims
    # Type A: One +/-2, rest 0
    for i in range(8):
        for s in (2, -2):
            v = [0] * D
            v[i] = s
            vectors.append(v)
            
    # Type B: Four +/-1, rest 0
    for positions in itertools.combinations(range(8), 4):
        for mask in range(16):
            v = [0] * D
            for idx, pos in enumerate(positions):
                if (mask >> idx) & 1:
                    v[pos] = -1
                else:
                    v[pos] = 1
            vectors.append(v)
            
    if time.time() > deadline:
        return np.array(vectors, dtype=np.int64)

    # 2. Generate D3 integer vectors of norm squared 4 in last 3 dims
    # Type C: One +/-2, rest 0
    for i in range(3):
        for s in (2, -2):
            v = [0] * D
            v[8 + i] = s
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
