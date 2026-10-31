import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Generate all weight-4 vectors
    # Indices 0..10
    # Choose 4 indices
    # Assign signs +/-
    pool = []
    indices = range(D)
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            pool.append(v)
            
    pool = np.array(pool, dtype=np.int64)
    n_pool = len(pool)
    
    # Greedy selection
    selected = []
    selected_vecs = np.array([], dtype=np.int64).reshape(0, D)
    
    # Shuffle pool to avoid bias?
    # For now, just iterate.
    # To improve, we can do multiple restarts or random shuffles.
    
    # Optimization: Precompute norms? All are 2.
    # Condition: u . v <= 2
    
    for i in range(n_pool):
        if time.time() > deadline:
            break
            
        v = pool[i]
        
        if len(selected_vecs) == 0:
            selected.append(i)
            selected_vecs = v.reshape(1, D)
            continue
            
        # Compute dot products with all selected vectors
        dots = selected_vecs @ v
        
        if np.max(dots) <= 2:
            selected.append(i)
            selected_vecs = np.vstack([selected_vecs, v.reshape(1, D)])
            
    # If we have time, try to improve with random restarts or different orders
    # But 5280 is small enough that a single greedy run might be good.
    # However, greedy order matters.
    
    # Let's try a few random shuffles if time permits
    best_set = selected_vecs
    best_len = len(selected_vecs)
    
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline:
                break
            np.random.shuffle(pool)
            selected = []
            selected_vecs = np.array([], dtype=np.int64).reshape(0, D)
            
            for i in range(n_pool):
                if time.time() > deadline:
                    break
                v = pool[i]
                if len(selected_vecs) == 0:
                    selected_vecs = v.reshape(1, D)
                    continue
                dots = selected_vecs @ v
                if np.max(dots) <= 2:
                    selected_vecs = np.vstack([selected_vecs, v.reshape(1, D)])
                    
            if len(selected_vecs) > best_len:
                best_len = len(selected_vecs)
                best_set = selected_vecs
                
    return best_set
# EVOLVE-BLOCK-END
