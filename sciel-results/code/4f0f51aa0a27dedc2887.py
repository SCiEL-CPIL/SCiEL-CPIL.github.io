"""Improved portfolio: Better ALS with structured initialization for 2x5x5 matrix multiplication tensor."""

import time
import numpy as np
import sys

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # 145 seconds safety margin (must be < 150s)
    
    n, m, p = 2, 5, 5
    
    # Helper to build schoolbook
    def build_schoolbook(rank):
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # Verify a decomposition
    def is_valid(U, V, W, tol=1e-8):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(total - 1) > tol:
                        return False
        return True
    
    # Compute error for a decomposition
    def compute_error(U, V, W):
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(total - 1))
        return max_err
    
    # Start with schoolbook as best known
    best_rank = n * m * p
    U_best, V_best, W_best = build_schoolbook(best_rank)
    
    # Try to find lower rank decompositions using optimization
    def try_rank(R, seed=42):
        """Try to find a valid decomposition of rank R using optimization."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        # Structured initialization: start from schoolbook and perturb
        # Or use a smarter initialization
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        
        # Scale to reasonable magnitude
        U /= np.max(np.abs(U)) + 1e-10
        V /= np.max(np.abs(V)) + 1e-10
        W /= np.max(np.abs(W)) + 1e-10
        
        # Alternating least squares with better convergence
        for it in range(200):
            if time.time() > deadline - 5:
                return None
            
            # Solve for U given V and W
            # For each (i,j): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # This is: U[i*m+j, :] @ diag(V[j*p+k, :] * W[k*n+i, :]) = 1 for each k
            # Actually, for fixed (i,j), we have p equations in R unknowns
            # Let's build the system properly
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Variables: U[i*m+j, 0..R-1]
                    # Constraints: for each k in 0..p-1:
                    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    A_sub = np.zeros((p, R))
                    for k in range(p):
                        A_sub[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b_sub = np.ones(p)
                    try:
                        U_new[i*m+j, :] = np.linalg.lstsq(A_sub, b_sub, rcond=None)[0]
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            U = U_new
            
            # Solve for V given U and W
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # Variables: V[j*p+k, 0..R-1]
                    # Constraints: for each i in 0..n-1:
                    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    A_sub = np.zeros((n, R))
                    for i in range(n):
                        A_sub[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b_sub = np.ones(n)
                    try:
                        V_new[j*p+k, :] = np.linalg.lstsq(A_sub, b_sub, rcond=None)[0]
                    except:
                        V_new[j*p+k, :] = V[j*p+k, :]
            V = V_new
            
            # Solve for W given U and V
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # Variables: W[k*n+i, 0..R-1]
                    # Constraints: for each j in 0..m-1:
                    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    A_sub = np.zeros((m, R))
                    for j in range(m):
                        A_sub[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b_sub = np.ones(m)
                    try:
                        W_new[k*n+i, :] = np.linalg.lstsq(A_sub, b_sub, rcond=None)[0]
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            W = W_new
            
            # Check convergence
            err = compute_error(U, V, W)
            if err < 1e-7:
                return U, V, W
        
        return None
    
    # Try ranks from just below 50 downward with multiple seeds
    for R in range(best_rank - 1, max(best_rank - 15, 35), -1):
        if time.time() > deadline - 10:
            break
        found = False
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            result = try_rank(R, seed=seed + 100)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = R
                found = True
                break
        if found:
            break
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
