import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0

    # Schoolbook solution as fallback
    rank_sb = n * m * p  # 84
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Precompute equation indices
    # Equations indexed by (i, k, i2, k2, j, j2)
    eq_i, eq_k, eq_i2, eq_k2, eq_j, eq_j2 = np.meshgrid(
        range(n), range(p), range(n), range(p), range(m), range(m),
        indexing='ij'
    )
    eq_i = eq_i.ravel()
    eq_k = eq_k.ravel()
    eq_i2 = eq_i2.ravel()
    eq_k2 = eq_k2.ravel()
    eq_j = eq_j.ravel()
    eq_j2 = eq_j2.ravel()

    num_eqs = len(eq_i)  # 3*7*3*7*4*4 = 7056

    # Precompute b
    b = np.zeros(num_eqs)
    b[(eq_i == eq_i2) & (eq_k == eq_k2) & (eq_j == eq_j2)] = 1.0

    # Precompute W position: k*n + i
    w_col_offset = eq_k * n + eq_i  # shape (num_eqs,)

    # Precompute U and V row indices
    u_row_idx = eq_i2 * m + eq_j  # shape (num_eqs,)
    v_row_idx = eq_j2 * p + eq_k2  # shape (num_eqs,)

    # For each equation e and rank index r, the W column that gets the coefficient
    # We need: A[e, r*21 + w_col_offset[e]] = U[e,r] * V[e,r]
    # This is a sparse structure. Let's build it efficiently.
    
    # Precompute the column indices for each (e, r) pair
    # col_idx[e, r] = r * 21 + w_col_offset[e]
    col_idx = np.zeros((num_eqs, 1), dtype=np.int64)  # will expand
    # Actually, let's just build A column by column as before, but vectorized differently.
    
    # Alternative: build A as a sparse-like operation
    # For each r, A[:, r*21 + w_col_offset] = U[u_row_idx, r] * V[v_row_idx, r]
    # This is: for each e, A[e, r*21 + w_col_offset[e]] = coeff[e, r]
    
    # Let's precompute the full column index matrix
    # col_indices shape (num_eqs, R) where col_indices[e, r] = r*21 + w_col_offset[e]
    # We'll build this on the fly for each R.
    
    # Search for lower rank
    # Known: matrix multiplication 3x4 by 4x7
    # Let's try ranks from 83 down to 70
    for R in range(rank_sb - 1, 69, -1):
        if time.time() > deadline - 15.0:
            break
        
        max_trials = 300
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline - 15.0:
                break
            
            # Random integer entries in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n * m, R))
            V = np.random.randint(-1, 2, size=(m * p, R))
            
            # Build A vectorized
            # A shape: (num_eqs, 21*R)
            A = np.zeros((num_eqs, 21 * R))
            
            U_vals = U[u_row_idx, :]  # (num_eqs, R)
            V_vals = V[v_row_idx, :]  # (num_eqs, R)
            coeffs = U_vals * V_vals  # (num_eqs, R)
            
            # Scatter: for each r, place coeffs[:, r] at column r*21 + w_col_offset
            # We can do this with advanced indexing
            for r in range(R):
                A[:, r * 21 + w_col_offset] = coeffs[:, r]
            
            # Solve least squares
            try:
                w, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            if len(residuals) > 0 and residuals[0] > 1e-9:
                continue
            
            # Check if w is half-integer (or integer)
            w2 = w * 2
            w2_rounded = np.round(w2)
            if np.allclose(w2, w2_rounded, atol=1e-9):
                W_sol = w.reshape((21, R))
                if R < best_rank:
                    best_rank = R
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W_sol.copy()
                    found = True
                    break
        
        if found:
            # Continue to find lower rank
            pass
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
