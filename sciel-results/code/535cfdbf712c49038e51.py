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
    
    def compute_residual_fast(U, V, W):
        """Compute residual vector using vectorized operations."""
        R = U.shape[1]
        residuals = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m+j, :]
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    residuals[idx] = np.sum(u_row * v_row * w_row) - 1.0
                    idx += 1
        return residuals
    
    def max_abs_residual(U, V, W):
        return np.max(np.abs(compute_residual_fast(U, V, W)))
    
    def als_optimize(U, V, W, max_iter=1000, tol=1e-12, reg=1e-6):
        """Alternating Least Squares optimization for tensor decomposition."""
        R = U.shape[1]
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            old_err = max_abs_residual(U, V, W)
            if old_err < tol:
                break
            
            # Update U: for each row u_idx, solve for U[u_idx,:]
            for i in range(n):
                for j in range(m):
                    u_idx = i*m+j
                    # Collect all (j,k) pairs where U[u_idx, r] appears
                    # For fixed i,j: residual involves V[j*p+k, r] * W[k*n+i, r] for k=0..p-1
                    A = np.zeros((p, R))
                    b = np.zeros(p)
                    for k in range(p):
                        v_row = V[j*p+k, :]
                        w_row = W[k*n+i, :]
                        A[k, :] = v_row * w_row
                        b[k] = 1.0
                    # Solve: A @ U[u_idx,:] = b
                    try:
                        U[u_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each row v_idx, solve for V[v_idx,:]
            for j in range(m):
                for k in range(p):
                    v_idx = j*p+k
                    A = np.zeros((n, R))
                    b = np.zeros(n)
                    for i in range(n):
                        u_row = U[i*m+j, :]
                        w_row = W[k*n+i, :]
                        A[i, :] = u_row * w_row
                        b[i] = 1.0
                    try:
                        V[v_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each row w_idx, solve for W[w_idx,:]
            for k in range(p):
                for i in range(n):
                    w_idx = k*n+i
                    A = np.zeros((m, R))
                    b = np.zeros(m)
                    for j in range(m):
                        u_row = U[i*m+j, :]
                        v_row = V[j*p+k, :]
                        A[j, :] = u_row * v_row
                        b[j] = 1.0
                    try:
                        W[w_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            new_err = max_abs_residual(U, V, W)
            if abs(old_err - new_err) < tol:
                break
        
        return U, V, W
    
    def gauss_newton_polish(U, V, W, max_iter=100, tol=1e-14):
        """High-precision Gauss-Newton polish for final refinement."""
        R = U.shape[1]
        total_vars = U.size + V.size
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            res = compute_residual_fast(U, V, W)
            max_res = np.max(np.abs(res))
            if max_res < tol:
                break
            
            # Build Jacobian (sparse structure exploited)
            J = np.zeros((n*m*p, total_vars))
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i*m+j
                        v_idx = j*p+k
                        w_idx = k*n+i
                        for r in range(R):
                            # d/dU[u_idx, r]
                            J[idx, u_idx*R + r] = V[v_idx, r] * W[w_idx, r]
                            # d/dV[v_idx, r]
                            J[idx, U.size + v_idx*R + r] = U[u_idx, r] * W[w_idx, r]
                        idx += 1
            
            # Solve (J^T J) delta = -J^T res
            try:
                JtJ = J.T @ J
                JtJ += 1e-10 * np.eye(total_vars)
                Jtres = J.T @ res
                delta = np.linalg.solve(JtJ, -Jtres)
            except np.linalg.LinAlgError:
                try:
                    delta = np.linalg.lstsq(JtJ, -Jtres, rcond=None)[0]
                except:
                    break
            
            # Update U and V only (W determined by constraint)
            U_flat = U.flatten()
            V_flat = V.flatten()
            U_flat += delta[:U.size]
            V_flat += delta[U.size:]
            
            U = U_flat.reshape(U.shape)
            V = V_flat.reshape(V.shape)
            
            # Update W using the constraint
            for k in range(p):
                for i in range(n):
                    w_idx = k*n+i
                    for r in range(R):
                        # W[w_idx, r] = (1 - sum_{r'!=r} U[i*m+j,r']*V[j*p+k,r']*W[w_idx,r']) / (U[i*m+j,r]*V[j*p+k,r])
                        # This is tricky; instead just keep W fixed in polish
                        pass
        
        return U, V, W
    
    def try_rank_als(R, seed=42, n_inits=5):
        """Try to find valid decomposition of rank R using ALS."""
        if time.time() > deadline - 15:
            return None
        
        best_result = None
        best_err = float('inf')
        
        for init_iter in range(n_inits):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed + init_iter * 13)
            
            # Structured initialization: use schoolbook pattern as base
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            if R >= n*m*p:
                # Schoolbook initialization
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i*m+j, col] = 1
                            V[j*p+k, col] = 1
                            W[k*n+i, col] = 1
                            col += 1
            else:
                # Random initialization with scaling
                U = np.random.randn(n*m, R) * 0.5
                V = np.random.randn(m*p, R) * 0.5
                W = np.random.randn(p*n, R) * 0.5
                
                # Normalize columns to unit max
                for r in range(R):
                    U[:, r] /= (np.max(np.abs(U[:, r])) + 1e-10)
                    V[:, r] /= (np.max(np.abs(V[:, r])) + 1e-10)
                    W[:, r] /= (np.max(np.abs(W[:, r])) + 1e-10)
            
            # ALS optimization
            U, V, W = als_optimize(U, V, W, max_iter=500, tol=1e-12)
            
            err = max_abs_residual(U, V, W)
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                # High-precision polish
                U, V, W = gauss_newton_polish(U, V, W, max_iter=50, tol=1e-14)
                err = max_abs_residual(U, V, W)
                if err < 1e-12:
                    return U, V, W
        
        if best_result is not None and best_err < 1e-9:
            U, V, W = best_result
            # Final polish
            U, V, W = gauss_newton_polish(U, V, W, max_iter=100, tol=1e-14)
            if max_abs_residual(U, V, W) < 1e-10:
                return U, V, W
        
        return None
    
    # Start with schoolbook
    best_rank = n * m * p
    U_best, V_best, W_best = build_schoolbook(best_rank)
    
    # Try ranks from known lower bound upward
    # For 2x5x5, lower bound is around 40-41
    for R in range(40, 51):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            result = try_rank_als(R, seed=seed * 7 + 1, n_inits=3)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = R
                found = True
                break
        
        if found:
            # Try to go lower
            for R2 in range(R-1, 35, -1):
                if time.time() > deadline - 15:
                    break
                result = try_rank_als(R2, seed=100 + R2, n_inits=5)
                if result is not None:
                    U_best, V_best, W_best = result
                    best_rank = R2
                    break
            break
    
    # Final high-precision polish
    if best_rank < n * m * p:
        U_best, V_best, W_best = gauss_newton_polish(U_best, V_best, W_best, max_iter=200, tol=1e-15)
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
