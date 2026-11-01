"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Helper to validate a decomposition
    def is_valid_decomposition(U, V, W, rank):
        U_mat = np.array(U)
        V_mat = np.array(V)
        W_mat = np.array(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i*m+j, :]
                    v_val = V_mat[j*p+k, :]
                    w_val = W_mat[k*n+i, :]
                    total = np.sum(u_val * v_val * w_val)
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        return True

    # 1. Establish Baseline (Schoolbook)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_sb}
    
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    # Search for lower rank
    # Lower bound heuristic: max(n*m, m*p, p*n) is not a lower bound for rank, but rank >= max(n, m, p) is trivial.
    # For 2x5x6, rank is likely around 50.
    # We search from a reasonable lower bound up to schoolbook - 1.
    
    min_rank = 40 # Start a bit lower than expected to be safe, or just go down from schoolbook?
    # Going down is better if we expect a specific rank.
    # But finding rank 50 when schoolbook is 60 is a 10% reduction.
    # Let's try ranks from schoolbook - 1 down to min_rank.
    
    found_better = False
    
    for r in range(rank_sb - 1, min_rank - 1, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Number of attempts per rank
        # If r is close to schoolbook, it should be easy.
        # If r is much lower, it might be hard.
        attempts = 500
        
        for _ in range(attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Generate random sparse integer U, V
            # Density: Let's try 0.3 or 0.5
            # Entries in {-1, 0, 1}
            
            U_rand = np.zeros((n * m, r), dtype=int)
            V_rand = np.zeros((m * p, r), dtype=int)
            
            # Randomly fill with -1, 0, 1
            # To ensure solvability, we need enough non-zeros.
            # Let's try a specific pattern or just random.
            
            mask_u = np.random.rand(n * m, r) < 0.4
            mask_v = np.random.rand(m * p, r) < 0.4
            
            U_rand[mask_u] = np.random.choice([-1, 1], size=mask_u.sum())
            V_rand[mask_v] = np.random.choice([-1, 1], size=mask_v.sum())
            
            # Construct Linear System for W
            # Variables: W[k*n + i, c] for k in 0..p-1, i in 0..n-1, c in 0..r-1
            # Total vars: p * n * r
            num_vars = p * n * r
            num_eqs = n * m * p
            
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            # Fill A
            # Eq index: (i, j, k) -> i*m*p + j*p + k
            # Var index: (k, i, c) -> (k*n + i) * r + c
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        # Coeff for W[k, i, c] is U[i, j, c] * V[j, k, c]
                        # U row: i*m + j
                        # V row: j*p + k
                        u_row = U_rand[i * m + j]
                        v_row = V_rand[j * p + k]
                        coeffs = u_row * v_row # shape (r,)
                        
                        # Var indices for this (k, i)
                        base_var_idx = (k * n + i) * r
                        A[eq_idx, base_var_idx : base_var_idx + r] = coeffs
            
            # Solve A W_vec = b
            # Use least squares
            try:
                W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
                
            # Check if W_vec entries are close to integers or half-integers
            # Multiply by 2 to check if they are integers
            W_check = W_vec * 2
            W_rounded = np.round(W_check)
            
            # Check distance
            diff = np.abs(W_check - W_rounded)
            if np.max(diff) < 1e-6:
                # Potential solution
                W_snap = (W_rounded / 2.0)
                
                # Convert back to matrices
                W_mat = W_snap.reshape(p * n, r)
                
                # Verify exact decomposition with snapped W
                if is_valid_decomposition(U_rand.tolist(), V_rand.tolist(), W_mat.tolist(), r):
                    best_result = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_mat.tolist(), "rank": r}
                    found_better = True
                    break
                    
        if found_better:
            break
        if time.time() - start_time > time_limit:
            break
            
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
# EVOLVE-BLOCK-END
