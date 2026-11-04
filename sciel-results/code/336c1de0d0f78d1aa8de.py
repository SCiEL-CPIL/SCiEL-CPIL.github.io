"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
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
    
    # Precompute index arrays for fast verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    def check_decomposition(U, V, W, rank, tol=1e-8):
        """Fast vectorized check of decomposition."""
        U_arr = np.asarray(U, dtype=np.float64)
        V_arr = np.asarray(V, dtype=np.float64)
        W_arr = np.asarray(W, dtype=np.float64)
        
        total = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank):
            total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        
        return np.max(np.abs(total - 1.0)) <= tol
    
    def compute_residual(U, V, W, rank):
        """Compute residual for optimization."""
        U_arr = np.asarray(U, dtype=np.float64)
        V_arr = np.asarray(V, dtype=np.float64)
        W_arr = np.asarray(W, dtype=np.float64)
        
        total = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank):
            total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        
        return total - 1.0
    
    # Value set for exactness
    value_set = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    # Strategy: Try ranks from low to high with structured initialization
    # Use block structure: 3x3x8 can be seen as 3 blocks of 3x1x8
    
    for rank in range(24, 72):
        if time.time() > deadline - 10:
            break
        
        max_attempts = max(10, min(50, 200 - rank * 2))
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Structured random init based on block structure
            U = np.zeros((n * m, rank), dtype=np.float64)
            V = np.zeros((m * p, rank), dtype=np.float64)
            W = np.zeros((p * n, rank), dtype=np.float64)
            
            # For each of the 3 "j" blocks, assign some rank dimensions
            rank_per_block = rank // 3
            remaining = rank - rank_per_block * 3
            
            for j in range(m):
                block_start = j * rank_per_block + min(j, remaining)
                block_size = rank_per_block + (1 if j < remaining else 0)
                
                for r in range(block_start, block_start + block_size):
                    # Random assignment within the block
                    for i in range(n):
                        if random.random() < 0.6:
                            U[i * m + j, r] = random.choice(value_set)
                    for k in range(p):
                        if random.random() < 0.6:
                            V[j * p + k, r] = random.choice(value_set)
                    for i in range(n):
                        if random.random() < 0.6:
                            W[k * n + i, r] = random.choice(value_set)
            
            # Check if valid
            if check_decomposition(U, V, W, rank, tol=1e-6):
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                best_rank = rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                return best_result
            
            # Local search polish
            current_residual = np.max(np.abs(compute_residual(U, V, W, rank)))
            
            if current_residual < 1.0:
                for polish_iter in range(30):
                    if time.time() > deadline - 5:
                        break
                    
                    mat_choice = random.choice(['U', 'V', 'W'])
                    if mat_choice == 'U':
                        mat = U
                        rows = n * m
                    elif mat_choice == 'V':
                        mat = V
                        rows = m * p
                    else:
                        mat = W
                        rows = p * n
                    
                    row = random.randint(0, rows - 1)
                    col = random.randint(0, rank - 1)
                    
                    old_val = mat[row, col]
                    best_val = old_val
                    best_res = current_residual
                    
                    for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0, -1.0]:
                        mat[row, col] = old_val + delta
                        res = np.max(np.abs(compute_residual(U, V, W, rank)))
                        if res < best_res:
                            best_res = res
                            best_val = mat[row, col]
                    
                    mat[row, col] = best_val
                    current_residual = best_res
                    
                    if current_residual < 1e-8:
                        break
                
                if current_residual < 1e-6:
                    # Snap to nearest value
                    U_snapped = np.zeros_like(U)
                    V_snapped = np.zeros_like(V)
                    W_snapped = np.zeros_like(W)
                    for mat, snapped in [(U, U_snapped), (V, V_snapped), (W, W_snapped)]:
                        for i in range(len(mat)):
                            for j in range(len(mat[0])):
                                closest = min(value_set, key=lambda v: abs(v - mat[i, j]))
                                snapped[i, j] = closest
                    
                    if check_decomposition(U_snapped, V_snapped, W_snapped, rank, tol=1e-6):
                        U_list = [[float(x) for x in row] for row in U_snapped]
                        V_list = [[float(x) for x in row] for row in V_snapped]
                        W_list = [[float(x) for x in row] for row in W_snapped]
                        best_rank = rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
