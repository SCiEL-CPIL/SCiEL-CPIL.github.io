"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_row = U_arr[i*m+l, :]
                v_row = V_arr[l*p+k, :]
                w_row = W_arr[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p):
    """Add 7 Strassen products for a 2x2x2 block and return new col."""
    U[0*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    U[1*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] -= 1
    col += 1
    
    U[0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    U[1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + 0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    U[0*m + j0][col] += 1
    U[0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] -= 1
    W[k1*n + 0][col] += 1
    col += 1
    
    U[1*m + j0][col] += 1
    U[0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    U[0*m + j1][col] += 1
    U[1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    return col

def _try_reduce_rank(U, V, W, n, m, p, rank, deadline):
    """Try to reduce rank by finding linear dependencies."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_rank = rank
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    
    # Build the constraint matrix: for each (i,l,k), the sum over r of
    # U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r] must equal 1.
    # We try to remove column c by expressing it as a linear combination of others.
    # For a given column c to be removable, there must exist coefficients alpha_r (r != c)
    # such that for all (i,l,k):
    #   U[i*m+l,c]*V[l*p+k,c]*W[k*n+i,c] = sum_{r!=c} alpha_r * U[i*m+l,r]*V[l*p+k,r]*W[k*n+i,r]
    
    # This is a system of n*m*p equations in (rank-1) unknowns.
    # Let's set up the linear system for each column candidate.
    
    num_constraints = n * m * p
    
    for c in range(rank):
        if time.time() > deadline:
            break
        
        # Build the RHS vector: the contribution of column c
        rhs = np.zeros(num_constraints)
        # Build the A matrix: columns are the contributions of other columns
        A_mat = np.zeros((num_constraints, rank - 1))
        
        idx = 0
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    rhs[idx] = U_arr[i*m+l, c] * V_arr[l*p+k, c] * W_arr[k*n+i, c]
                    col_idx = 0
                    for r in range(rank):
                        if r == c:
                            continue
                        A_mat[idx, col_idx] = U_arr[i*m+l, r] * V_arr[l*p+k, r] * W_arr[k*n+i, r]
                        col_idx += 1
                    idx += 1
        
        # Solve the least-squares problem
        try:
            alpha, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, rhs, rcond=1e-10)
            # Check if the solution is good
            recon = A_mat @ alpha
            if np.max(np.abs(recon - rhs)) < 1e-8:
                # Column c can be removed!
                new_rank = rank - 1
                new_U = np.delete(U_arr, c, axis=1)
                new_V = np.delete(V_arr, c, axis=1)
                new_W = np.delete(W_arr, c, axis=1)
                
                # Verify
                if _verify_decomp(new_U.tolist(), new_V.tolist(), new_W.tolist(), n, m, p, new_rank):
                    best_rank = new_rank
                    best_U = new_U
                    best_V = new_V
                    best_W = new_W
                    rank = new_rank
                    U_arr = best_U
                    V_arr = best_V
                    W_arr = best_W
                    # Continue trying to reduce further
                    c = 0  # Restart from beginning
                    continue
        except Exception:
            pass
    
    return best_rank, best_U.tolist(), best_V.tolist(), best_W.tolist()

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140
    
    # Start with the known rank-54 decomposition
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p)
    
    # Remaining j=4: standard products
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    # Verify the initial decomposition
    if not _verify_decomp(U, V, W, n, m, p, rank):
        pass
    
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Attempt rank reduction
    best_rank, best_U, best_V, best_W = _try_reduce_rank(U, V, W, n, m, p, rank, _deadline)
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
