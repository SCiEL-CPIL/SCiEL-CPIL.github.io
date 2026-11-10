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

    def greedy_select(pool, max_dot, time_limit, num_restarts=50):
        n = len(pool)
        if n == 0:
            return np.empty((0, d), dtype=np.int64)
            
        best_set_indices = []
        
        for r in range(num_restarts):
            if time.time() > time_limit:
                break
                
            np.random.seed(42 + r)
            order = np.random.permutation(n)
            
            selected_indices = []
            selected_matrix = np.empty((0, d), dtype=np.int64)
            
            for idx in order:
                if time.time() > time_limit:
                    break
                
                v = pool[idx]
                
                if len(selected_indices) == 0:
                    selected_indices.append(idx)
                    selected_matrix = v.reshape(1, -1)
                    continue
                
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
    
    # Try higher weights
    weights_to_try = [7, 8, 9, 10, 11]
    
    # Calculate max_dot for each weight
    # Norm^2 = w
    # 2w - 2 dot >= w => 2 dot <= w => dot <= w/2
    # Integer dot: floor(w/2)
    
    for w in weights_to_try:
        if time.time() > deadline - 10:
            break
            
        max_dot = w // 2
        pool = generate_pool(w)
        
        # Adjust restarts based on pool size and remaining time
        # Smaller pools (w=11) can have more restarts
        # Larger pools (w=7) need fewer restarts to finish in time
        # But we want to maximize k.
        
        # Estimate time per restart.
        # Let's just use a fixed number and rely on the time check inside.
        # For w=7, pool is ~42k. Greedy is O(K^2). If K=1000, 1M ops.
        # 50 restarts = 50M ops. Should be fast.
        
        res = greedy_select(pool, max_dot, deadline, num_restarts=100)
        
        if len(res) > best_k:
            best_k = len(res)
            best_result = res
            
    # Fallback to weight-2 if all else fails
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
