"""Rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication."""

import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 2, 5, 5
    # Tensor indices: (i,j) for U row, (j,k) for V row, (k,i) for W row
    # Constraint: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
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
    
    def compute_residual(U, V, W):
        """Compute the residual vector of all constraints."""
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    residuals.append(val - 1.0)
        return np.array(residuals)
    
    def max_abs_residual(U, V, W):
        return np.max(np.abs(compute_residual(U, V, W)))
    
    def als_step(U, V, W, max_iter=100, tol=1e-10):
        """Alternating Least Squares to refine the decomposition."""
        R = U.shape[1]
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Solve for U
            # For each row u_idx in U, we have constraints:
            # sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r] = 1
            # This is a linear system for U[u_idx, :]
            # Let X_u = U[u_idx, :]
            # We need to find X_u such that for all (j,k) consistent with u_idx=i*m+j:
            # dot(X_u, V[j*p+k, :] * W[k*n+i, :]) = 1
            
            # Actually, it's easier to solve the full system for U given V, W.
            # The system is:
            # For each (i,j,k):
            # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
            # This is a linear system A_U * U_flat = b
            # Where A_U has rows corresponding to (i,j,k) and columns to U elements.
            # But U elements are shared across different (i,j,k) if they have the same (i,j).
            # So we can solve for each row of U independently? No, the RHS is 1 for all.
            # The coefficient for U[i*m+j, r] in the equation for (i,j,k) is V[j*p+k, r] * W[k*n+i, r].
            # Since the equation is for a specific (i,j,k), and U row is (i,j),
            # we can solve for each row of U independently using the equations involving that row.
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    u_idx = i*m+j
                    # Equations for this u_idx:
                    # For each k in range(p):
                    # sum_r U[u_idx, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    # Let M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # We need U[u_idx, :] such that M @ U[u_idx, :] = 1 (vector of ones of length p)
                    M = np.zeros((p, R))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        # Solve M @ x = ones(p)
                        U_new[u_idx, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        U_new[u_idx, :] = U[u_idx, :]
            
            # Solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    v_idx = j*p+k
                    # Equations for this v_idx:
                    # For each i in range(n):
                    # sum_r U[i*m+j, r] * V[v_idx, r] * W[k*n+i, r] = 1
                    # Let M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    # We need V[v_idx, :] such that M @ V[v_idx, :] = 1 (vector of ones of length n)
                    M = np.zeros((n, R))
                    for i in range(n):
                        M[i, :] = U_new[i*m+j, :] * W[k*n+i, :]
                    try:
                        V_new[v_idx, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        V_new[v_idx, :] = V[v_idx, :]
            
            # Solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    w_idx = k*n+i
                    # Equations for this w_idx:
                    # For each j in range(m):
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[w_idx, r] = 1
                    # Let M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    # We need W[w_idx, :] such that M @ W[w_idx, :] = 1 (vector of ones of length m)
                    M = np.zeros((m, R))
                    for j in range(m):
                        M[j, :] = U_new[i*m+j, :] * V_new[j*p+k, :]
                    try:
                        W_new[w_idx, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        W_new[w_idx, :] = W[w_idx, :]
            
            U, V, W = U_new, V_new, W_new
            
            err = max_abs_residual(U, V, W)
            if err < tol:
                break
        
        return U, V, W

    def try_rank(R, seed=42):
        """Try to find valid decomposition of rank R."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        best_result = None
        best_err = float('inf')
        
        # Multiple initializations
        for init_iter in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n*m, R) * 0.3
            V = np.random.randn(m*p, R) * 0.3
            W = np.random.randn(p*n, R) * 0.3
            
            # ALS refinement
            U, V, W = als_step(U, V, W, max_iter=100, tol=1e-10)
            
            err = max_abs_residual(U, V, W)
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-9:
                return U, V, W
        
        if best_result is not None and best_err < 1e-8:
            return best_result
        return None

    # Start with schoolbook
    best_rank = n * m * p
    U_best, V_best, W_best = build_schoolbook(best_rank)
    
    # Try ranks from 40 upward (known lower bounds suggest ~40 for 2x5x5)
    # Actual known rank is 38.
    for R in range(38, 45):
        if time.time() > deadline - 15:
            break
        found = False
        for seed in range(50):
            if time.time() > deadline - 8:
                break
            result = try_rank(R, seed=seed * 7 + 1)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = R
                found = True
                break
        if found:
            break
    
    # Final polish
    if best_rank < n * m * p:
        U_best, V_best, W_best = als_step(U_best, V_best, W_best, max_iter=200, tol=1e-12)
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
