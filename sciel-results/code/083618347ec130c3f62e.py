"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    n_rows = n * m
    m_rows = m * p
    p_rows = p * n
    rank = U.shape[1]
    
    if U.shape != (n_rows, rank) or V.shape != (m_rows, rank) or W.shape != (p_rows, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j, :]
                v_val = V[j * p + k, :]
                w_val = W[k * n + i, :]
                product_sum = np.sum(u_val * v_val * w_val)
                if abs(product_sum - 1.0) > tol:
                    return False
    return True

def _build_linear_system(n, m, p, R):
    """
    Build the linear system A x = b for a fixed rank R.
    Unknowns: R columns, each with n*m + m*p + p*n entries.
    Total unknowns: R * (n*m + m*p + p*n)
    
    For each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
    This is nonlinear. But if we fix U and V, it's linear in W, and vice versa.
    
    Actually, the standard approach: fix one matrix, solve for the other two.
    But that's still nonlinear jointly.
    
    Alternative: Use the fact that for matrix multiplication, there's a known
    lower bound. For 4x4 * 4x8, the rank is at least max(2*4*4, 2*4*8, 4*4*8/...) 
    Actually, the bilinear rank for n×m times m×p is at least max(nm, mp, np) and 
    at most nmp. Known results: for square n×n, rank is between n^2 and n^3.
    For 4x4x8, let's try to find a good rank.
    
    Let's use an iterative approach: start with a random initialization at rank R,
    then use alternating least squares or similar to converge.
    """
    pass

def _als_decompose(n, m, p, R, max_iter=200, tol=1e-12, seed=42):
    """
    Alternating Least Squares for tensor decomposition of matrix multiplication.
    T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all valid (i,j,k).
    
    We minimize the error ||T - sum_c U_c ⊗ V_c ⊗ W_c||_F subject to the constraint
    that all entries equal 1.
    """
    np.random.seed(seed)
    
    U = np.random.randn(n * m, R) * 0.1
    V = np.random.randn(m * p, R) * 0.1
    W = np.random.randn(p * n, R) * 0.1
    
    for iteration in range(max_iter):
        # Update U: for each (i,j), find U[i*m+j,:] to minimize error
        for i in range(n):
            for j in range(m):
                # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # Let A[k,c] = V[j*p+k,c] * W[k*n+i,c] for k=0..p-1, c=0..R-1
                # Then we need A @ U[i*m+j,:] = 1 (vector of ones of length p)
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve A @ u = 1
                u, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U[i * m + j, :] = u
        
        # Update V: for each (j,k), find V[j*p+k,:]
        for j in range(m):
            for k in range(p):
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                v, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                V[j * p + k, :] = v
        
        # Update W: for each (k,i), find W[k*n+i,:]
        for k in range(p):
            for i in range(n):
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                w, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                W[k * n + i, :] = w
        
        # Check convergence
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < tol:
            break
    
    return U, V, W, max_err

def _try_rank(n, m, p, R, max_iter=200, seed=42, tol=1e-12):
    """Try to find a decomposition of rank R."""
    U, V, W, err = _als_decompose(n, m, p, R, max_iter=max_iter, tol=tol, seed=seed)
    if err < 1e-9:
        return U, V, W, True
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 128
    
    # Try to find minimal rank
    # Lower bound: for matrix multiplication, rank >= max(n*m, m*p, p*n)
    # = max(16, 32, 32) = 32
    # But actually, the bilinear rank for n×m × m×p is at least nm + mp + np - 1 in some cases
    # Let's try from a reasonable lower bound up
    
    # Known: 4x4x4 has rank at most 49 (Strassen-like), but we don't have those constants
    # Let's try ranks from 48 upward (since 4*4*8=128, and we want to be below that)
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = schoolbook_rank
    
    # Start with schoolbook
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try to find lower ranks
    # Lower bound estimate: at least max(n*m, m*p, p*n) = 32
    # Let's try from 48 down to 32, or up from 32
    # Actually, let's try a range of ranks
    
    ranks_to_try = list(range(48, 128))  # Try from 48 up to 127
    
    # Sort by likelihood of success - try lower ranks first but with more iterations
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        if R >= best_rank:
            continue
        
        # Try multiple seeds for this rank
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            U_trial, V_trial, W_trial, success = _try_rank(
                n, m, p, R, max_iter=300, seed=seed, tol=1e-12
            )
            
            if success:
                # Verify
                if _verify_decomposition(U_trial, V_trial, W_trial, n, m, p, tol=1e-9):
                    best_U, best_V, best_W = U_trial.copy(), V_trial.copy(), W_trial.copy()
                    best_rank = R
                    print(f"Found valid rank {best_rank} with seed {seed}")
                    break  # Move to next lower rank
        
        # If we found a rank, continue searching for lower ones
        if best_rank <= R:
            continue
    
    # If no better rank found, use schoolbook
    if best_rank == schoolbook_rank:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
    else:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
