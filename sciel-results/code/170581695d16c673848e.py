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
        
        # For each rank component, add contribution
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
    
    # Try to find a decomposition using structured random search + local polish
    # Value set: use small integers and halves for exactness
    value_set = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    # Strategy: Try ranks from low to high, with multiple random restarts
    # For each rank, do random init then local search to reduce residual
    
    for rank in range(24, 72):
        if time.time() > deadline - 10:
            break
        
        # Number of attempts: more for lower ranks (harder)
        max_attempts = max(20, min(80, 300 - rank * 3))
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.zeros((n * m, rank), dtype=np.float64)
            V = np.zeros((m * p, rank), dtype=np.float64)
            W = np.zeros((p * n, rank), dtype=np.float64)
            
            # Use structured random init: ensure some columns are "active"
            density = random.uniform(0.4, 0.8)
            
            for i in range(n * m):
                for r in range(rank):
                    if random.random() < density:
                        U[i, r] = random.choice(value_set)
            
            for i in range(m * p):
                for r in range(rank):
                    if random.random() < density:
                        V[i, r] = random.choice(value_set)
            
            for i in range(p * n):
                for r in range(rank):
                    if random.random() < density:
                        W[i, r] = random.choice(value_set)
            
            # Check if already valid
            if check_decomposition(U, V, W, rank, tol=1e-6):
                # Convert to list of lists with exact values
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                best_rank = rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                return best_result
            
            # Local search: try to reduce residual by adjusting entries
            # Use gradient-free local search with small perturbations
            current_residual = np.max(np.abs(compute_residual(U, V, W, rank)))
            
            if current_residual < 0.5:  # Close enough to try polishing
                # Try local search for up to 50 iterations
                for polish_iter in range(50):
                    if time.time() > deadline - 5:
                        break
                    
                    # Pick a random matrix and entry to perturb
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
                    
                    # Try small perturbations
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
                
                # Check if polished solution is valid
                if current_residual < 1e-6:
                    # Snap to nearest value in value_set for exactness
                    U_snapped = np.zeros_like(U)
                    V_snapped = np.zeros_like(V)
                    W_snapped = np.zeros_like(W)
                    for mat, snapped in [(U, U_snapped), (V, V_snapped), (W, W_snapped)]:
                        for i in range(len(mat)):
                            for j in range(len(mat[0])):
                                # Find closest value in value_set
                                closest = min(value_set, key=lambda v: abs(v - mat[i, j]))
                                snapped[i, j] = closest
                    
                    if check_decomposition(U_snapped, V_snapped, W_snapped, rank, tol=1e-6):
                        U_list = [[float(x) for x in row] for row in U_snapped]
                        V_list = [[float(x) for x in row] for row in V_snapped]
                        W_list = [[float(x) for x in row] for row in W_snapped]
                        best_rank = rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        return best_result
    
    # Fallback: return schoolbook
    return best_result
# EVOLVE-BLOCK-END
