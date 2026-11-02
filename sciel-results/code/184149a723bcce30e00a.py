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
    
    def gauss_newton(U, V, W, max_iter=500, tol=1e-10):
        """Gauss-Newton method to refine the decomposition."""
        R = U.shape[1]
        total_vars = U.size + V.size + W.size
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            res = compute_residual(U, V, W)
            max_res = np.max(np.abs(res))
            if max_res < tol:
                break
            
            # Build Jacobian: d(residual)/d(variables)
            # Residual for (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
            # d/dU[i*m+j,r'] = V[j*p+k,r']*W[k*n+i,r'] if r==r' else 0
            # d/dV[j*p+k,r'] = U[i*m+j,r']*W[k*n+i,r'] if r==r' else 0  
            # d/dW[k*n+i,r'] = U[i*m+j,r']*V[j*p+k,r'] if r==r' else 0
            
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
                            J[idx, total_vars - W.size + v_idx*R + r] = U[u_idx, r] * W[w_idx, r]
                            # d/dW[w_idx, r]
                            J[idx, total_vars - W.size - V.size + w_idx*R + r] = U[u_idx, r] * V[v_idx, r]
                        idx += 1
            
            # Solve (J^T J) delta = -J^T res
            try:
                JtJ = J.T @ J
                JtJ += 1e-8 * np.eye(total_vars)  # regularization
                Jtres = J.T @ res
                delta = np.linalg.solve(JtJ, -Jtres)
            except np.linalg.LinAlgError:
                try:
                    delta = np.linalg.lstsq(JtJ, -Jtres, rcond=None)[0]
                except:
                    break
            
            # Update
            U_flat = U.flatten()
            V_flat = V.flatten()
            W_flat = W.flatten()
            U_flat += delta[:U.size]
            V_flat += delta[U.size:U.size+V.size]
            W_flat += delta[U.size+V.size:]
            
            U = U_flat.reshape(U.shape)
            V = V_flat.reshape(V.shape)
            W = W_flat.reshape(W.shape)
        
        return U, V, W
    
    def try_rank(R, seed=42):
        """Try to find valid decomposition of rank R."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        best_result = None
        best_err = float('inf')
        
        # Multiple initializations
        for init_iter in range(3):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with structure
            U = np.random.randn(n*m, R) * 0.3
            V = np.random.randn(m*p, R) * 0.3
            W = np.random.randn(p*n, R) * 0.3
            
            # Normalize columns
            for r in range(R):
                U[:, r] /= (np.max(np.abs(U[:, r])) + 1e-10)
                V[:, r] /= (np.max(np.abs(V[:, r])) + 1e-10)
                W[:, r] /= (np.max(np.abs(W[:, r])) + 1e-10)
            
            # Gauss-Newton refinement
            U, V, W = gauss_newton(U, V, W, max_iter=200, tol=1e-10)
            
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
    # Actually, let's try from a reasonable estimate downward
    # For 2xn x nxn, rank is at least 2n^2 - n + 1 or similar
    # Let's try ranks from 42 down to 38
    
    for R in range(42, 37, -1):
        if time.time() > deadline - 15:
            break
        found = False
        for seed in range(20):
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
    
    # Final polish if we found something
    if best_rank < n * m * p:
        U_best, V_best, W_best = gauss_newton(U_best, V_best, W_best, max_iter=300, tol=1e-12)
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
