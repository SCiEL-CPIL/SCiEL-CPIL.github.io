import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    candidates = []
    
    # Type 1: 2*E8
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = 2*si, 2*sj
                candidates.append(v)
    for signs in itertools.product([1, -1], repeat=8):
        v = [0] * D
        for k in range(8):
            v[k] = signs[k]
        candidates.append(v)
        
    # Type 2: Last 3
    for w in itertools.product([-2, 2, 0], repeat=3):
        if sum(x*x for x in w) == 8:
            v = [0] * 8 + list(w)
            candidates.append(v)
            
    # Type 3: 4+4
    u4 = []
    for idxs in itertools.combinations(range(8), 4):
        for signs in itertools.product([1, -1], repeat=4):
            u = [0] * 8
            for k, idx in enumerate(idxs):
                u[idx] = signs[k]
            u4.append(u)
    for idx in range(8):
        for s in (1, -1):
            u = [0] * 8
            u[idx] = 2*s
            u4.append(u)
            
    w4 = []
    for idx in range(3):
        for s in (1, -1):
            w = [0] * 3
            w[idx] = 2*s
            w4.append(w)
            
    for u in u4:
        for w in w4:
            candidates.append(u + w)
            
    # Type 4: 2+6
    u2 = []
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                u = [0] * 8
                u[i], u[j] = si, sj
                u2.append(u)
                
    w6 = set()
    for idxs in itertools.permutations(range(3), 3):
        for signs in itertools.product([1, -1], repeat=3):
            w = [0] * 3
            w[idxs[0]] = 2*signs[0]
            w[idxs[1]] = 1*signs[1]
            w[idxs[2]] = 1*signs[2]
            if sum(x*x for x in w) == 6:
                w6.add(tuple(w))
    w6 = [list(w) for w in w6]
    
    for u in u2:
        for w in w6:
            candidates.append(u + w)
            
    # Type 5: 3+5
    u3 = []
    for idxs in itertools.combinations(range(8), 3):
        for signs in itertools.product([1, -1], repeat=3):
            u = [0] * 8
            for k, idx in enumerate(idxs):
                u[idx] = signs[k]
            u3.append(u)
            
    w5 = set()
    for idxs in itertools.permutations(range(3), 2):
        for s2 in (1, -1):
            for s1 in (1, -1):
                w = [0] * 3
                w[idxs[0]] = 2*s2
                w[idxs[1]] = 1*s1
                if sum(x*x for x in w) == 5:
                    w5.add(tuple(w))
    w5 = [list(w) for w in w5]
    
    for u in u3:
        for w in w5:
            candidates.append(u + w)

    candidates = np.array(candidates, dtype=np.int64)
    np.random.shuffle(candidates)
    
    selected = []
    selected_arr = np.empty((0, D), dtype=np.int64)
    
    for v in candidates:
        if time.time() > deadline:
            break
        if len(selected_arr) == 0:
            selected.append(v)
            selected_arr = selected_arr.reshape(0, D) # just for logic
            selected_arr = v.reshape(1, D)
        else:
            dots = selected_arr @ v
            if np.max(dots) <= 4:
                selected.append(v)
                selected_arr = np.vstack([selected_arr, v.reshape(1, D)])
                
    return np.array(selected, dtype=np.int64)
