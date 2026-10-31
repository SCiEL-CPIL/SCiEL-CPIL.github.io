import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    D = 11
    indices = range(D)
    
    # Generate all weight-4 vectors in Z^11
    # C(11,4) * 2^4 = 330 * 16 = 5280 vectors
    pool = []
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            pool.append(v)
    
    pool = np.array(pool, dtype=np.int8)
    n_pool = len(pool)  # 5280
    
    # For weight-4 vectors, norm^2 = 4.
    # Condition: ||u - v||^2 >= max norm = 4 (since all have norm^2=4)
    # ||u-v||^2 = 4 + 4 - 2*dot(u,v) = 8 - 2*dot(u,v) >= 4
    # 2*dot(u,v) <= 4 => dot(u,v) <= 2
    # So conflict if dot(u,v) >= 3
    
    dots = pool @ pool.T
    # conflict_mask[i,j] = True if vectors i and j cannot both be selected
    conflict_mask = dots >= 3
    
    best_set = np.array([], dtype=np.int64).reshape(0, D)
    best_len = 0
    
    # Strategy: Min-degree greedy with multiple random perturbations
    # For each restart, we use a randomized greedy:
    # - Pick a random active vertex, add it, remove its neighbors
    # - This is fast with vectorized operations on 5280 elements
    
    num_restarts = 500  # will be limited by time
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        active = np.ones(n_pool, dtype=bool)
        current_selected = []
        
        while np.any(active):
            if time.time() > deadline - 2:
                break
            
            # Pick a random active vertex
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            v = active_indices[np.random.randint(len(active_indices))]
            
            current_selected.append(v)
            
            # Remove v and all its neighbors from active
            active[v] = False
            # Remove neighbors: active &= ~conflict_mask[v]
            active = active & ~conflict_mask[v]
        
        current_len = len(current_selected)
        if current_len > best_len:
            best_len = current_len
            best_set = pool[current_selected]
    
    return best_set
