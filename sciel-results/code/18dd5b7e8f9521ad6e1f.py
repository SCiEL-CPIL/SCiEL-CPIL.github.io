"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

_start_time = time.time()
_deadline = _start_time + 110

def construct():
    """
    Returns a rank-minimal tensor decomposition for 3x3 * 3x6 matrix multiplication.
    Tries to find a decomposition with rank < 54 using random search and linear algebra.
    Falls back to schoolbook rank-54 if no better solution is found.
    """
    n, m, p = 3, 3, 6
    
    # Schoolbook rank-54 decomposition
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
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Convert to numpy for faster operations
    U_s = np.array(U_school, dtype=float)
    V_s = np.array(V_school, dtype=float)
    W_s = np.array(W_school, dtype=float)
    
    # Target tensor T (all ones)
    T = np.ones((n, m, p), dtype=float)
    
    # Try to find a lower rank solution
    # We search from rank_school - 1 down to some minimum
    start_time = time.time()
    deadline = 105.0  # Safety margin
    
    # Strategy: Try random rank-r decompositions
    # For each rank r, we try to find U, V, W such that sum_r u_r * v_r * w_r = T
    # We can fix U and V randomly and solve for W linearly
    
    np.random.seed(42)
    
    for target_rank in range(rank_school - 1, max(rank_school - 15, 40), -1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        max_trials = 20 if target_rank > 50 else 10
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
                
            # Generate random U and V matrices with entries from {-1, 0, 1}
            U_rand = np.zeros((n * m, target_rank), dtype=float)
            V_rand = np.zeros((m * p, target_rank), dtype=float)
            
            for idx in range(n * m * target_rank):
                U_rand[idx // target_rank, idx % target_rank] = np.random.choice([-1, 0, 1])
            for idx in range(m * p * target_rank):
                V_rand[idx // target_rank, idx % target_rank] = np.random.choice([-1, 0, 1])
            
            # Set up linear system for W
            # Variables: W_{k,i,r} for k in 0..p-1, i in 0..n-1, r in 0..target_rank-1
            # Total variables: p * n * target_rank
            num_vars = p * n * target_rank
            
            # Matrix A (size n*m*p x num_vars)
            # Row index: (i, j, k) -> i*m*p + j*p + k
            # Col index: (k, i, r) -> k*n*target_rank + i*target_rank + r
            
            A = np.zeros((n * m * p, num_vars), dtype=float)
            b = np.zeros(n * m * p, dtype=float)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = i * m * p + j * p + k
                        b[row] = 1.0
                        for r in range(target_rank):
                            col_idx = k * n * target_rank + i * target_rank + r
                            A[row, col_idx] = U_rand[i * m + j, r] * V_rand[j * p + k, r]
            
            # Solve A W' = b
            try:
                W_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            # Check if solution is close to integer/half-integer
            W_flat = W_sol.flatten()
            W_rounded = np.round(W_flat * 2) / 2.0
            
            # Check error
            W_rounded_matrix = W_rounded.reshape(p, n, target_rank)
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += U_rand[i * m + j, r] * V_rand[j * p + k, r] * W_rounded_matrix[k, i, r]
                        error += abs(val - 1.0)
            
            if error < 1e-9:
                # Found a valid solution
                U_new = U_rand.tolist()
                V_new = V_rand.tolist()
                W_new = W_rounded_matrix.tolist()
                
                return {"U": U_new, "V": V_new, "W": W_new}
            
            # If not exact, we can't use it directly
            # But maybe we can adjust? For now, just continue searching
    
    # If no lower rank found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
