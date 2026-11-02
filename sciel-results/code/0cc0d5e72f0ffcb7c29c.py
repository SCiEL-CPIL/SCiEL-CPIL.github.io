import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = 110.0
    
    # 1. Generate D11 roots (Norm 2)
    d11_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                d11_vectors.append(tuple(v))
                
    # 2. Generate 2e_i (Norm 4)
    two_e_vectors = []
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            two_e_vectors.append(tuple(v))
            
    # 3. Generate Four 1s (Norm 4)
    four_ones_vectors = []
    indices = range(D)
    for combo in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            four_ones_vectors.append(tuple(v))
            
    # 4. Greedy Independent Set on Four 1s
    # Conflict if dot > 2
    best_set = []
    
    # Optimization: Pre-compute? No, just check on fly.
    # 5280 vectors.
    # Greedy:
    for _ in range(20): # 20 restarts
        if time.time() > deadline: break
        random.shuffle(four_ones_vectors)
        current_set = []
        current_np = [] # List of tuples or arrays
        
        for v in four_ones_vectors:
            valid = True
            for u in current_set:
                # Dot product
                dot = 0
                for k in range(D):
                    if v[k] != 0 and u[k] != 0:
                        dot += v[k] * u[k]
                if dot > 2:
                    valid = False
                    break
            if valid:
                current_set.append(v)
                
        if len(current_set) > len(best_set):
            best_set = current_set
            
    # Combine
    all_vectors = d11_vectors + two_e_vectors + best_set
    
    return np.array(all_vectors, dtype=np.int64)
