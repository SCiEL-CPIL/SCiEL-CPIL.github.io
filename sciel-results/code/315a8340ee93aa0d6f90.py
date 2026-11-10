import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    d = 11
    
    def generate_pool(w):
        pool = []
        for idx in itertools.combinations(range(d), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = np.zeros(d, dtype=np.int64)
                for i, s in zip(idx, signs):
                    v[i] = s
                pool.append(v)
        return np.array(pool)

    def greedy_select(pool, max_dot, time_limit, num_restarts=20):
        n = len(pool)
        if n == 0:
            return np.empty((0, d), dtype=np.int64)
            
        best_set_indices = []
        
        np.random.seed(42) # Reproducibility for the first restart, then vary
        
        for r in range(num_restarts):
            if time.time() > time_limit:
                break
                
            # Vary seed for subsequent restarts
            np.random.seed(42 + r)
            order = np.random.permutation(n)
            
            selected_vectors = []
            selected_indices = []
            
            # Optimization: Pre-allocate if possible? No, list append is fine.
            # We will convert to array at the end of the pass to check size, 
            # but for the loop, we need fast dot products.
            # Keeping a list of vectors and converting to array every time is slow.
            # Better: Keep a matrix `S` and append rows? 
            # Appending to a 2D numpy array is slow.
            # Strategy: Keep list of indices. At each step, if we add, 
            # we need to update the "selected matrix".
            # To avoid O(K) copy on every add, we can just keep the list of indices
            # and compute dots against the *current* selected vectors.
            # But computing dots against a list is slow in Python.
            
            # Efficient approach:
            # Keep `selected_indices` as a list.
            # Maintain `selected_matrix` as a numpy array.
            # When adding, `selected_matrix = np.vstack([selected_matrix, pool[idx]])`.
            # This is O(K) per addition. Total O(K^2) per pass.
            # For K=500, 250k ops. Fast.
            
            selected_matrix = np.empty((0, d), dtype=np.int64)
            
            for idx in order:
                if time.time() > time_limit:
                    break
                
                v = pool[idx]
                
                if len(selected_indices) == 0:
                    selected_indices.append(idx)
                    selected_matrix = v.reshape(1, -1)
                    continue
                
                # Check compatibility
                dots = selected_matrix @ v
                if np.max(dots) <= max_dot:
                    selected_indices.append(idx)
                    selected_matrix = np.vstack([selected_matrix, v.reshape(1, -1)])
            
            if len(selected_indices) > len(best_set_indices):
                best_set_indices = selected_indices[:]
                
        if not best_set_indices:
             return np.empty((0, d), dtype=np.int64)
             
        return pool[best_set_indices]

    best_result = None
    best_k = 0
    
    # Try w=4
    if time.time() < deadline:
        pool4 = generate_pool(4)
        res4 = greedy_select(pool4, 2, deadline, num_restarts=30)
        if len(res4) > best_k:
            best_k = len(res4)
            best_result = res4

    # Try w=5
    if time.time() < deadline - 60: # Reserve time for w=6
        pool5 = generate_pool(5)
        # For w=5, pool is larger. Reduce restarts if time is tight?
        # Let's try 15 restarts.
        res5 = greedy_select(pool5, 2, deadline, num_restarts=15)
        if len(res5) > best_k:
            best_k = len(res5)
            best_result = res5

    # Try w=6
    if time.time() < deadline - 30:
        pool6 = generate_pool(6)
        res6 = greedy_select(pool6, 3, deadline, num_restarts=10)
        if len(res6) > best_k:
            best_k = len(res6)
            best_result = res6

    # Fallback to w=2 if all else fails (shouldn't happen)
    if best_result is None or best_k < 220:
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for si in (-1, 1):
                    for sj in (-1, 1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        points.append(v)
        best_result = np.array(points, dtype=np.int64)
        best_k = len(best_result)

    return np.around(best_result).astype(np.int64)
