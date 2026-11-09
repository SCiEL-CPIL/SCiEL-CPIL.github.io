"""Program to find a low-rank exact decomposition for 3x3 * 3x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import nnls
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    margin = 10.0
    
    n, m, p = 3, 3, 5
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # U shape: (n*m, rank) = (9, rank)
    # V shape: (m*p, rank) = (15, rank)
    # W shape: (p*n, rank) = (15, rank)
    
    def schoolbook_decomposition():
        """Return rank-45 schoolbook decomposition."""
        rank = n * m * p  # 45
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition is exact."""
        rank = U.shape[1]
        # For each (i,j,k), check sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    max_err = max(max_err, abs(s - 1.0))
        return max_err < tol
    
    def round_to_half_integers(U, V, W):
        """Round entries to nearest half-integers."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r
    
    def alternating_minimization(rank, max_iters=200, tol=1e-8, seed=42):
        """
        Alternating least squares for tensor decomposition.
        Fix two factors, solve for the third via linear least squares.
        """
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Normalize to avoid scale issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - margin:
                break
            
            # Solve for U: fix V and W
            # For each (i,j), we need: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
            # This is: U[i*m+j,:] @ diag(V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
            # Actually: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Let M_k = V[j*p+k,:] * W[k*n+i,:] (element-wise product)
            # Then sum_r U[i*m+j,r] * M_k[r] = 1
            # This is a linear system: U[i*m+j,:] @ M^T = 1^T where M is p x rank
            # More precisely: for fixed (i,j), we have p equations in rank unknowns (U[i*m+j,:])
            # A @ U[i*m+j,:] = b where A[k,r] = V[j*p+k,r] * W[k*n+i,r], b[k] = 1
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Build system: A @ x = b
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    # Solve least squares
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = x
            
            # Solve for V: fix U and W
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
                    # A[i,r] = U[i*m+j,r] * W[k*n+i,r]
                    # A @ V[j*p+k,:] = 1
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = x
            
            # Solve for W: fix U and V
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
                    # A[j,r] = U[i*m+j,r] * V[j*p+k,r]
                    # A @ W[k*n+i,:] = 1
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = x
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            if check_decomposition(U, V, W, tol=1e-4):
                # Try rounding
                U_r, V_r, W_r = round_to_half_integers(U, V, W)
                if check_decomposition(U_r, V_r, W_r, tol=1e-4):
                    return U_r, V_r, W_r, it
                else:
                    return U, V, W, it
            
            err = check_decomposition(U, V, W, tol=float('inf'))
            if abs(err - prev_err) < 1e-10:
                break
            prev_err = err
        
        # Final check
        if check_decomposition(U, V, W, tol=1e-4):
            U_r, V_r, W_r = round_to_half_integers(U, V, W)
            if check_decomposition(U_r, V_r, W_r, tol=1e-4):
                return U_r, V_r, W_r, max_iters
            else:
                return U, V, W, max_iters
        return None, None, None, max_iters
    
    # Try to find low-rank decompositions
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 45
    
    # Try ranks from 37 up to 44
    for rank in range(37, min(45, best_rank)):
        if time.time() > deadline - margin:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(20):
            if time.time() > deadline - margin:
                break
            
            U_try, V_try, W_try, iters = alternating_minimization(rank, max_iters=100, seed=seed)
            if U_try is not None and check_decomposition(U_try, V_try, W_try, tol=1e-4):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = rank
                found = True
                break
        
        if found:
            break
    
    # If we found something, convert to list of lists
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Fallback: Schoolbook
    U, V, W = schoolbook_decomposition()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
