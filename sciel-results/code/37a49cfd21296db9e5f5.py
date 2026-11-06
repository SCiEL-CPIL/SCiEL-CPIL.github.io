import numpy as np
import time
import scipy.sparse as sp

def construct():
    N = 256
    start_time = time.time()
    time_limit = 55.0
    
    # Generate APs
    row_indices = []
    col_indices = []
    aps_0 = []
    
    for d in range(1, N):
        for a in range(1, N + 1):
            indices = list(range(a - 1, N, d))
            if len(indices) < 2:
                continue
            aps_0.append(indices)
            for i in indices:
                row_indices.append(len(aps_0) - 1)
                col_indices.append(i)
                
    num_aps = len(aps_0)
    
    data = np.ones(len(row_indices))
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_aps, N))
    
    affected_by = [[] for _ in range(N)]
    for idx, indices in enumerate(aps_0):
        for i in indices:
            affected_by[i].append(idx)
            
    rng = np.random.default_rng(42)
    
    def local_search(x):
        x = x.copy()
        sums = A @ x
        current_max = np.max(np.abs(sums))
        
        # Pre-compute abs sums for speed? No, sums change.
        
        while time.time() - start_time < time_limit:
            abs_sums = np.abs(sums)
            max_idx = np.argmax(abs_sums)
            worst_ap = aps_0[max_idx]
            
            # Try flipping a random element in the worst AP
            i = rng.integers(0, len(worst_ap))
            pos = worst_ap[i]
            
            old_val = x[pos]
            x[pos] = -old_val
            delta = -2 * old_val
            
            affected = affected_by[pos]
            if affected:
                sums[affected] += delta
            
            new_max = np.max(np.abs(sums))
            
            if new_max < current_max:
                current_max = new_max
            else:
                x[pos] = old_val
                if affected:
                    sums[affected] -= delta
                    
            # Random perturbation to escape local minima
            if rng.random() < 0.1:
                pos_rand = rng.integers(0, N)
                old_val = x[pos_rand]
                x[pos_rand] = -old_val
                delta = -2 * old_val
                affected = affected_by[pos_rand]
                if affected:
                    sums[affected] += delta
                new_max = np.max(np.abs(sums))
                if new_max < current_max:
                    current_max = new_max
                else:
                    x[pos_rand] = old_val
                    if affected:
                        sums[affected] -= delta
                        
        return x, current_max

    best_x = None
    best_max_val = float('inf')
    
    # Initial guesses
    # 1. Random
    # 2. Alternating
    # 3. Random
    
    seeds = []
    seeds.append(rng.choice([-1, 1], size=N))
    seeds.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)]))
    seeds.append(rng.choice([-1, 1], size=N))
    
    for x_init in seeds:
        if time.time() - start_time > time_limit - 5:
            break
        x_res, max_val = local_search(x_init)
        if max_val < best_max_val:
            best_max_val = max_val
            best_x = x_res
            
    # If time remains, do more random restarts
    while time.time() - start_time < time_limit - 2:
        x_init = rng.choice([-1, 1], size=N)
        x_res, max_val = local_search(x_init)
        if max_val < best_max_val:
            best_max_val = max_val
            best_x = x_res
            
    if best_x is None:
        return list(rng.choice([-1, 1], size=N))
        
    return list(best_x)

construct()
