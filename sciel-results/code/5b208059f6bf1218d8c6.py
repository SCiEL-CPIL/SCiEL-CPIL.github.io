import numpy as np
import time
import random

def construct():
    n, m, p = 3, 3, 3
    deadline = time.time() + 110  # Safety margin
    
    # Indices for the 27 non-zero entries of the tensor T
    # T[i, j, k] = 1 if it contributes to C[i, k]
    # Equation index e corresponds to (i, j, k)
    # Variable index for W is (c, k*n + i)
    # Coefficient is U[i*m+j, c] * V[j*p+k, c]
    
    eq_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_indices.append((i, j, k))
                
    num_eqs = len(eq_indices)
    
    best_rank = 27
    best_U, best_V, best_W = None, None, None
    
    # Seed with naive solution
    U_naive = np.zeros((n*m, 27), dtype=int)
    V_naive = np.zeros((m*p, 27), dtype=int)
    W_naive = np.zeros((p*n, 27), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i*m+j, col] = 1
                V_naive[j*p+k, col] = 1
                W_naive[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_naive, V_naive, W_naive
    best_rank = 27
    
    # Search for lower rank
    # Known minimum is 23. We try 22 (impossible) to 26.
    for rank in range(22, 27):
        if time.time() > deadline:
            break
            
        # Try more trials with mixed sparse and dense random matrices
        # Dense {-1, 1} is more likely to produce a well-conditioned M
        # such that the solution Y = 2W is integer.
        
        # Number of trials depends on remaining time
        # Estimate time per trial: ~1ms? 1000 trials = 1s
        # 5 ranks * 1000 trials = 5s. Safe.
        num_trials = 1000
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            # Alternate between sparse and dense
            if trial % 2 == 0:
                # Sparse
                U = np.zeros((n*m, rank), dtype=int)
                V = np.zeros((m*p, rank), dtype=int)
                for c in range(rank):
                    for idx in range(n*m):
                        if random.random() < 0.3:
                            U[idx, c] = random.choice([-1, 1])
                    for idx in range(m*p):
                        if random.random() < 0.3:
                            V[idx, c] = random.choice([-1, 1])
            else:
                # Dense
                U = np.random.choice([-1, 1], size=(n*m, rank))
                V = np.random.choice([-1, 1], size=(m*p, rank))
                
            # Construct the linear system matrix M
            # Size: num_eqs x (p*n * rank)
            M = np.zeros((num_eqs, p*n * rank))
            
            for e_idx, (i, j, k) in enumerate(eq_indices):
                w_idx_base = k * n + i
                for c in range(rank):
                    coeff = U[i*m+j, c] * V[j*p+k, c]
                    if coeff != 0:
                        M[e_idx, c * p*n + w_idx_base] = coeff
            
            # Solve M * W_vec = 1
            # We want 2*W_vec to be integer.
            # Let Y = 2*W_vec. Then M * Y = 2 * ones
            try:
                # Use least squares to find Y
                Y_vec, residuals, rank_m, s = np.linalg.lstsq(M, 2 * np.ones(num_eqs), rcond=None)
                
                # Check if Y_vec is close to integers
                Y_rounded = np.round(Y_vec)
                error = np.max(np.abs(Y_vec - Y_rounded))
                
                if error < 1e-6:
                    # Valid solution found for Y
                    # W = Y / 2
                    W_solution = Y_rounded / 2.0
                    
                    # Verify the solution exactly
                    valid = True
                    for e_idx, (i, j, k) in enumerate(eq_indices):
                        w_idx = k * n + i
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W_solution[c, w_idx]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                            
                    if valid:
                        best_rank = rank
                        best_U = U
                        best_V = V
                        best_W = W_solution
                        break # Found best for this rank, break trial loop
            except Exception:
                continue
        
        if best_rank == rank:
            break # Found a solution for this rank, no need to check higher ranks (which are worse)
            
    # Convert to lists for return
    def to_list(mat):
        return mat.tolist()
        
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
