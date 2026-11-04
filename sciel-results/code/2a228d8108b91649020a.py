"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import random
    import time
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    schoolbook_rank = nm * p  # 42
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Build list of non-zero indices in T
    T_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_indices.append((i * m + j, j * p + k, k * n + i))
    
    # Sample zero indices for verification
    nonzero_set = set(T_indices)
    all_zero_indices = []
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                if (i, j, k) not in nonzero_set:
                    all_zero_indices.append((i, j, k))
    random.shuffle(all_zero_indices)
    T_zero_sample = all_zero_indices[:200]
    
    def compute_error(U, V, W):
        """Compute L1 error at non-zero and sampled zero entries."""
        r = U.shape[1]
        error = 0
        for (i, j, k) in T_indices:
            val = 0
            for c in range(r):
                val += U[i, c] * V[j, c] * W[k, c]
            error += abs(val - 1)
        for (i, j, k) in T_zero_sample:
            val = 0
            for c in range(r):
                val += U[i, c] * V[j, c] * W[k, c]
            if val != 0:
                error += abs(val)
        return error
    
    def full_verify(U, V, W):
        """Full verification of the decomposition."""
        r = U.shape[1]
        result = np.zeros((nm, mp, pn), dtype=np.int64)
        for c in range(r):
            result += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
        for (i, j, k) in T_indices:
            if result[i, j, k] != 1:
                return False
        for (i, j, k) in all_zero_indices:
            if result[i, j, k] != 0:
                return False
        return True
    
    def schoolbook():
        U = np.zeros((nm, schoolbook_rank), dtype=np.int64)
        V = np.zeros((mp, schoolbook_rank), dtype=np.int64)
        W = np.zeros((pn, schoolbook_rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = schoolbook()
    
    # Try to find lower rank using simulated annealing local search
    for target_rank in range(36, schoolbook_rank):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 15:
                break
            
            U = np.random.randint(-2, 3, (nm, target_rank), dtype=np.int64)
            V = np.random.randint(-2, 3, (mp, target_rank), dtype=np.int64)
            W = np.random.randint(-2, 3, (pn, target_rank), dtype=np.int64)
            
            current_error = compute_error(U, V, W)
            
            temp = 50.0
            for step in range(800):
                if time.time() > deadline - 15:
                    break
                if current_error == 0:
                    break
                
                factor_idx = random.randint(0, 2)
                if factor_idx == 0:
                    row = random.randint(0, nm - 1)
                    col = random.randint(0, target_rank - 1)
                    old_val = U[row, col]
                    U[row, col] = old_val + random.choice([-2, -1, 1, 2])
                    new_error = compute_error(U, V, W)
                    if new_error <= current_error or random.random() < np.exp(-(new_error - current_error) / max(temp, 1e-6)):
                        current_error = new_error
                    else:
                        U[row, col] = old_val
                elif factor_idx == 1:
                    row = random.randint(0, mp - 1)
                    col = random.randint(0, target_rank - 1)
                    old_val = V[row, col]
                    V[row, col] = old_val + random.choice([-2, -1, 1, 2])
                    new_error = compute_error(U, V, W)
                    if new_error <= current_error or random.random() < np.exp(-(new_error - current_error) / max(temp, 1e-6)):
                        current_error = new_error
                    else:
                        V[row, col] = old_val
                else:
                    row = random.randint(0, pn - 1)
                    col = random.randint(0, target_rank - 1)
                    old_val = W[row, col]
                    W[row, col] = old_val + random.choice([-2, -1, 1, 2])
                    new_error = compute_error(U, V, W)
                    if new_error <= current_error or random.random() < np.exp(-(new_error - current_error) / max(temp, 1e-6)):
                        current_error = new_error
                    else:
                        W[row, col] = old_val
                
                temp *= 0.998
            
            if current_error == 0 and full_verify(U, V, W):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break
        
        if not found:
            break
    
    U_list = [[int(x) for x in row] for row in best_U]
    V_list = [[int(x) for x in row] for row in best_V]
    W_list = [[int(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
