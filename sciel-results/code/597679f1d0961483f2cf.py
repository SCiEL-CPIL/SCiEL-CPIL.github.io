import numpy as np
import random
import itertools
import time

def construct():
    D = 11
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Build adjacency bitmask
    adj_mask = [0] * n_sup
    for i in range(n_sup):
        s_i = set(supports[i])
        for j in range(i+1, n_sup):
            s_j = set(supports[j])
            if len(s_i.intersection(s_j)) == 3:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
    def popcount(x):
        return x.bit_count()
        
    def is_compatible(v, S_mask):
        return (adj_mask[v] & S_mask) == 0

    def greedy_init():
        S_mask = 0
        order = list(range(n_sup))
        random.shuffle(order)
        for v in order:
            if is_compatible(v, S_mask):
                S_mask |= (1 << v)
        return S_mask

    best_mask = 0
    best_size = 0
    
    # Multiple greedy starts
    for _ in range(200):
        S_mask = greedy_init()
        size = popcount(S_mask)
        if size > best_size:
            best_size = size
            best_mask = S_mask
            
    # Simulated Annealing with Bitmasks
    S_mask = best_mask
    current_size = best_size
    T = 1.0
    cooling = 0.9995
    
    vertices = list(range(n_sup))
    
    start_time = time.time()
    timeout = 110 # Leave 10s for construction and overhead
    
    while time.time() - start_time < timeout:
        v = random.choice(vertices)
        
        if (S_mask >> v) & 1:
            continue # v is in S
            
        # Check if v can be added
        if is_compatible(v, S_mask):
            S_mask |= (1 << v)
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_mask = S_mask
        else:
            # v conflicts with S
            # Move: S -> S - N + {v} where N = neighbors(v) in S
            N_mask = adj_mask[v] & S_mask
            delta = 1 - popcount(N_mask)
            
            if delta > 0:
                S_mask = (S_mask & ~N_mask) | (1 << v)
                current_size += delta
            elif delta == 0:
                if random.random() < 0.5:
                    S_mask = (S_mask & ~N_mask) | (1 << v)
            else:
                p = np.exp(delta / T)
                if random.random() < p:
                    S_mask = (S_mask & ~N_mask) | (1 << v)
                    current_size += delta
                    
        T *= cooling
        
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask

    # Construct vectors
    vectors = []
    for idx in range(n_sup):
        if (best_mask >> idx) & 1:
            sup = supports[idx]
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for k, s in enumerate(sup):
                    v[s] = signs[k]
                vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
