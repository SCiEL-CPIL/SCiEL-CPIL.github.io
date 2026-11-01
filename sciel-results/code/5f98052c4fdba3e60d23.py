import numpy as np
import time
from fractions import Fraction
import sys

# Dimensions
n, m, p = 2, 3, 3
# Target rank to search for
TARGET_RANK = 17
# Schoolbook rank
SCHOOL_RANK = 18

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Target tensor T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p)
    # The tensor is the all-ones tensor of shape (2, 3, 3)
    # We seek U (shape 6xR), V (shape 9xR), W (shape 6xR) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    
    # 1. Construct Schoolbook Solution (Rank 18) as fallback
    U_school = np.zeros((n * m, SCHOOL_RANK))
    V_school = np.zeros((m * p, SCHOOL_RANK))
    W_school = np.zeros((p * n, SCHOOL_RANK))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, idx] = 1.0
                V_school[j * p + k, idx] = 1.0
                W_school[k * n + i, idx] = 1.0
                idx += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = SCHOOL_RANK
    
    # 2. Search for Rank 17 Solution
    # Strategy: Randomized ALS with multiple restarts, using high-precision verification.
    # We use a vectorized ALS update to be faster than the previous loop-based lstsq.
    
    max_restarts = 200
    max_iters = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        rng = np.random.RandomState(restart * 1000 + 42)
        
        # Initialize with schoolbook structure + noise for the first 17 terms, 
        # or fully random. Fully random often gets stuck in local minima.
        # A "perturbed schoolbook" approach: Take 17 of the 18 schoolbook terms, 
        # merge one pair using a parameter, and optimize.
        
        # Alternative: Start with random init but scale appropriately.
        U = rng.randn(n * m, TARGET_RANK)
        V = rng.randn(m * p, TARGET_RANK)
        W = rng.randn(p * n, TARGET_RANK)
        
        # Normalize to prevent divergence
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        prev_error = np.inf
        converged = False
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Vectorized ALS Update for U
            # For each (i, j), we solve U[i*m+j, :] @ (V[j*p+k,:] * W[k*n+i,:])_k = T[i,j,k]
            # This is a linear system: U_row @ M_ijk = T_ijk
            # Where M_ijk is a vector of length R formed by element-wise product of V row j and W row (k*n+i) for all k.
            
            # We can batch this.
            # Construct matrix A for all i, j? 
            # A[i, j, :, k] = V[j, :, k] * W[k, :, i] ? No, dimensions don't match simply.
            
            # Let's stick to the loop but vectorize the inner product part if possible, 
            # or just keep the loop but use a faster solver.
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Build matrix A (R x p) where A[r, k] = V[j*p+k, r] * W[k*n+i, r]
                    # A is (R, p). We want U[i*m+j, :] @ A = T[i, j, :] (1, p)
                    # This is (U_row (1xR)) @ A (RxP) = T_row (1xP)
                    # So A.T @ U_row.T = T_row.T
                    A = np.zeros((TARGET_RANK, p))
                    for k in range(p):
                        A[:, k] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve A.T @ u = T_col
                    # Use lstsq
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # A (R x n) where A[r, i] = U[i*m+j, r] * W[k*n+i, r]
                    # A.T @ v = T_col (n,)
                    A = np.zeros((TARGET_RANK, n))
                    for i in range(n):
                        A[:, i] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # A (R x m) where A[r, j] = U[i*m+j, r] * V[j*p+k, r]
                    # A.T @ w = T_col (m,)
                    A = np.zeros((TARGET_RANK, m))
                    for j in range(m):
                        A[:, j] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Compute Error
            # result[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Use einsum for speed
            U_r = U.reshape(n, m, TARGET_RANK)
            V_r = V.reshape(m, p, TARGET_RANK)
            W_r = W.reshape(p, n, TARGET_RANK)
            
            result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            error = np.linalg.norm(result - 1.0)
            
            if error < 1e-12:
                converged = True
                break
                
            if error > prev_error * 1.05 and it > 10:
                break
            prev_error = error
            
        if converged:
            # Verify and Round
            # Try rounding to nearest half-integer
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            # Exact verification with Fractions
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = Fraction(0)
                        for r in range(TARGET_RANK):
                            total += Fraction(U_int[i * m + j, r]).limit_denominator(10000) * \
                                       Fraction(V_int[j * p + k, r]).limit_denominator(10000) * \
                                       Fraction(W_int[k * n + i, r]).limit_denominator(10000)
                        if total != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_U, best_V, best_W = U_int, V_int, W_int
                best_rank = TARGET_RANK
                break # Found optimal, stop searching
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
