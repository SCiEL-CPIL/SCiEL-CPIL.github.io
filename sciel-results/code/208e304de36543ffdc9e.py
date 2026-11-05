import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    min_sum = 128
    max_sum = 256
    
    def compute_cost(A_vec):
        C = np.convolve(A_vec, A_vec, mode='full').astype(np.int32)
        
        D = np.zeros(257, dtype=np.int32)
        D[128:257:2] = A_vec[64:129]
        
        M = (C[128:257] + D[128:257]) // 2
        
        uncovered = np.sum(M == 0)
        if uncovered > 0:
            return 10000 + uncovered * 1000 + np.max(M)
            
        # Minimize max, then sum of squares for flatness
        max_m = np.max(M)
        sum_sq = np.sum(M**2)
        return max_m * 10000 + sum_sq

    def initial_solution():
        best_vec = None
        best_cost = float('inf')
        
        # Two blocks
        for a in range(20, 80, 2):
            for b in range(50, 110, 2):
                if b <= a:
                    continue
                vec = np.zeros(129, dtype=np.int32)
                vec[0:a+1] = 1
                vec[b:129] = 1
                cost = compute_cost(vec)
                if cost < best_cost:
                    best_cost = cost
                    best_vec = vec.copy()
                    
        # Single block
        for k in range(40, 80, 2):
            vec = np.zeros(129, dtype=np.int32)
            vec[k:129] = 1
            cost = compute_cost(vec)
            if cost < best_cost:
                best_cost = cost
                best_vec = vec.copy()
                
        # Random
        for size in range(40, 70, 5):
            for _ in range(20):
                vec = np.zeros(129, dtype=np.int32)
                idx = np.random.choice(129, size, replace=False)
                vec[idx] = 1
                cost = compute_cost(vec)
                if cost < best_cost:
                    best_cost = cost
                    best_vec = vec.copy()
                    
        if best_vec is None:
             vec = np.zeros(129, dtype=np.int32)
             vec[64:129] = 1
             best_vec = vec
             best_cost = compute_cost(vec)
             
        return best_vec, best_cost

    vec, current_cost = initial_solution()
    best_vec = vec.copy()
    best_cost = current_cost
    
    T = 5.0
    T_min = 0.01
    alpha = 0.9995
    
    random.seed(42)
    
    while time.time() < deadline:
        i = random.randint(0, 128)
        old_val = vec[i]
        vec[i] = 1 - old_val
        
        new_cost = compute_cost(vec)
        
        delta = new_cost - current_cost
        if delta < 0 or (T > T_min and random.random() < np.exp(-delta / T)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_vec = vec.copy()
        else:
            vec[i] = old_val
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    A_list = [i for i in range(129) if best_vec[i] == 1]
    
    if not A_list:
        return list(range(64, 129))
        
    return A_list
