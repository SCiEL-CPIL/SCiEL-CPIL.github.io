"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Precompute index arrays for fast evaluation
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    def check_decomposition_fast(U, V, W, rank):
        """Fast check using vectorized operations."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        total = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank):
            total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        
        if np.max(np.abs(total - 1.0)) > 1e-6:
            return False
        return True

    def local_search(rank, max_iters=20000):
        """
        Local search to find a decomposition of given rank.
        Uses incremental error update for speed.
        """
        # Initialize with random values in [-2, 2]
        U = np.random.randint(-2, 3, (n * m, rank)).astype(np.float64)
        V = np.random.randint(-2, 3, (m * p, rank)).astype(np.float64)
        W = np.random.randint(-2, 3, (p * n, rank)).astype(np.float64)
        
        # Compute initial S and Error
        # S[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        S = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank):
            S += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        
        error = np.sum((S - 1.0) ** 2)
        
        # Mapping from flat index to (i, j, k) and affected U/V/W indices
        # We need to update S efficiently.
        
        current_temp = 1.0
        min_temp = 0.01
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Pick a random matrix and entry
            mat_id = random.randint(0, 2)
            
            if mat_id == 0:
                row = random.randint(0, n * m - 1)
                col = random.randint(0, rank - 1)
                old_val = U[row, col]
                # Try perturbations
                delta = random.choice([-1, 1, -0.5, 0.5])
                new_val = old_val + delta
                if new_val < -3 or new_val > 3:
                    continue
                
                # Calculate change in S
                # S[i*m+j, j*p+k, k*n+i] is affected
                # row = i*m + j
                i = row // m
                j = row % m
                # Affected S indices: for all k, index = i*m*p + j*p + k? 
                # No, S is flattened as i*m*p + j*p + k?
                # My u_idx, v_idx, w_idx are based on a flat array of length n*m*p.
                # The flat index is idx = i * (m * p) + j * p + k.
                
                # We need to update S for all k in 0..p-1
                base_idx = i * (m * p) + j * p
                # Affected indices in S: base_idx + k for k in 0..p-1
                
                # Old contribution of this term to S:
                # Term = U[row, col] * V[j*p+k, col] * W[k*n+i, col]
                
                old_contrib = np.zeros(p)
                for k in range(p):
                    v_idx_val = j * p + k
                    w_idx_val = k * n + i
                    old_contrib[k] = old_val * V[v_idx_val, col] * W[w_idx_val, col]
                
                new_contrib = np.zeros(p)
                for k in range(p):
                    v_idx_val = j * p + k
                    w_idx_val = k * n + i
                    new_contrib[k] = new_val * V[v_idx_val, col] * W[w_idx_val, col]
                
                delta_S = new_contrib - old_contrib
                
                # Calculate delta error
                delta_error = 0.0
                for k in range(p):
                    s_old = S[base_idx + k]
                    s_new = s_old + delta_S[k]
                    delta_error += (s_new - 1.0)**2 - (s_old - 1.0)**2
                
                if delta_error < 0 or random.random() < np.exp(-delta_error / current_temp):
                    # Accept
                    U[row, col] = new_val
                    for k in range(p):
                        S[base_idx + k] += delta_S[k]
                    error += delta_error
                    if error < 1e-6:
                        # Check exactness
                        U_int = np.round(U * 2) / 2
                        V_int = np.round(V * 2) / 2
                        W_int = np.round(W * 2) / 2
                        if check_decomposition_fast(U_int, V_int, W_int, rank):
                            return U_int, V_int, W_int
                else:
                    # Reject
                    pass
                    
            elif mat_id == 1:
                row = random.randint(0, m * p - 1)
                col = random.randint(0, rank - 1)
                old_val = V[row, col]
                delta = random.choice([-1, 1, -0.5, 0.5])
                new_val = old_val + delta
                if new_val < -3 or new_val > 3:
                    continue
                
                # row = j*p + k
                j = row // p
                k = row % p
                # Affected S indices: for all i, index = i*m*p + j*p + k
                # base_idx depends on i.
                
                old_contrib = np.zeros(n)
                new_contrib = np.zeros(n)
                for i in range(n):
                    u_idx_val = i * m + j
                    w_idx_val = k * n + i
                    old_contrib[i] = U[u_idx_val, col] * old_val * W[w_idx_val, col]
                    new_contrib[i] = U[u_idx_val, col] * new_val * W[w_idx_val, col]
                
                delta_S = new_contrib - old_contrib
                
                delta_error = 0.0
                for i in range(n):
                    s_idx = i * (m * p) + j * p + k
                    s_old = S[s_idx]
                    s_new = s_old + delta_S[i]
                    delta_error += (s_new - 1.0)**2 - (s_old - 1.0)**2
                
                if delta_error < 0 or random.random() < np.exp(-delta_error / current_temp):
                    V[row, col] = new_val
                    for i in range(n):
                        S[i * (m * p) + j * p + k] += delta_S[i]
                    error += delta_error
                    if error < 1e-6:
                        U_int = np.round(U * 2) / 2
                        V_int = np.round(V * 2) / 2
                        W_int = np.round(W * 2) / 2
                        if check_decomposition_fast(U_int, V_int, W_int, rank):
                            return U_int, V_int, W_int
            else:
                row = random.randint(0, p * n - 1)
                col = random.randint(0, rank - 1)
                old_val = W[row, col]
                delta = random.choice([-1, 1, -0.5, 0.5])
                new_val = old_val + delta
                if new_val < -3 or new_val > 3:
                    continue
                
                # row = k*n + i
                k = row // n
                i = row % n
                # Affected S indices: for all j, index = i*m*p + j*p + k
                
                old_contrib = np.zeros(m)
                new_contrib = np.zeros(m)
                for j in range(m):
                    u_idx_val = i * m + j
                    v_idx_val = j * p + k
                    old_contrib[j] = U[u_idx_val, col] * V[v_idx_val, col] * old_val
                    new_contrib[j] = U[u_idx_val, col] * V[v_idx_val, col] * new_val
                
                delta_S = new_contrib - old_contrib
                
                delta_error = 0.0
                for j in range(m):
                    s_idx = i * (m * p) + j * p + k
                    s_old = S[s_idx]
                    s_new = s_old + delta_S[j]
                    delta_error += (s_new - 1.0)**2 - (s_old - 1.0)**2
                
                if delta_error < 0 or random.random() < np.exp(-delta_error / current_temp):
                    W[row, col] = new_val
                    for j in range(m):
                        S[i * (m * p) + j * p + k] += delta_S[j]
                    error += delta_error
                    if error < 1e-6:
                        U_int = np.round(U * 2) / 2
                        V_int = np.round(V * 2) / 2
                        W_int = np.round(W * 2) / 2
                        if check_decomposition_fast(U_int, V_int, W_int, rank):
                            return U_int, V_int, W_int
            
            # Cool down
            if it % 1000 == 0 and current_temp > min_temp:
                current_temp *= 0.9
        
        return None

    # Try ranks from 20 to 71
    for rank in range(20, 72):
        if time.time() > deadline - 5:
            break
        
        # Try a few attempts for each rank
        max_attempts = 5 if rank < 30 else 2
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            res = local_search(rank)
            if res:
                U_int, V_int, W_int = res
                best_rank = rank
                best_result = {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                return best_result

    return best_result
# EVOLVE-BLOCK-END
