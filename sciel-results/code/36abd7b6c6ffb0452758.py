import numpy as np
import time
import random
from scipy.optimize import least_squares

n, m, p = 2, 5, 6

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def check_decomposition(U, V, W, tol=1e-9):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def compute_tensor_residuals(U, V, W):
        """Compute all 60 residuals for the tensor."""
        R = U.shape[1]
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    res[idx] = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :])) - 1.0
                    idx += 1
        return res

    def alternating_least_squares(U, V, W, max_iters=200, tol=1e-12):
        """
        Alternating least squares for tensor rank factorization.
        Fix two factors, solve for the third.
        """
        R = U.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Solve for U: for each row u_idx, U[u_idx, :] * (V[v_idx,:] * W[w_idx,:])^T = 1
            # Group by u_idx
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    # Collect all (v_idx, w_idx) pairs that involve this u_idx
                    # v_idx = j*p + k, w_idx = k*n + i for k in range(p)
                    A_block = np.zeros((p, R))
                    b_block = np.ones(p)
                    for k in range(p):
                        v_idx = j * p + k
                        w_idx = k * n + i
                        A_block[k, :] = V[v_idx, :] * W[w_idx, :]
                    # Solve A_block^T @ U[u_idx,:] = b_block
                    try:
                        U_new[u_idx, :] = np.linalg.lstsq(A_block.T, b_block, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        U_new[u_idx, :] = U[u_idx, :]
            U = U_new
            
            # Solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    A_block = np.zeros((n, R))
                    b_block = np.ones(n)
                    for i in range(n):
                        u_idx = i * m + j
                        w_idx = k * n + i
                        A_block[i, :] = U[u_idx, :] * W[w_idx, :]
                    try:
                        V_new[v_idx, :] = np.linalg.lstsq(A_block.T, b_block, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        V_new[v_idx, :] = V[v_idx, :]
            V = V_new
            
            # Solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    A_block = np.zeros((m, R))
                    b_block = np.ones(m)
                    for j in range(m):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        A_block[j, :] = U[u_idx, :] * V[v_idx, :]
                    try:
                        W_new[w_idx, :] = np.linalg.lstsq(A_block.T, b_block, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        W_new[w_idx, :] = W[w_idx, :]
            W = W_new
            
            # Check convergence
            res = compute_tensor_residuals(U, V, W)
            if np.max(np.abs(res)) < tol:
                break
        
        return U, V, W

    def try_rank(R, n_restarts=5):
        """Try to find a rank-R decomposition using ALS with structured initializations."""
        if time.time() > deadline - 20:
            return None
        
        best_result = None
        best_err = np.inf
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Structured initialization: use a subset of schoolbook terms + noise
            U_s, V_s, W_s = get_schoolbook()
            
            if R >= 60:
                U = U_s.copy()
                V = V_s.copy()
                W = W_s.copy()
            else:
                # Pick R random columns from schoolbook, add small random perturbation
                cols = random.sample(range(60), R)
                U = U_s[:, cols].copy()
                V = V_s[:, cols].copy()
                W = W_s[:, cols].copy()
                # Add small random perturbation to escape degeneracy
                noise_scale = 0.1
                U += np.random.randn(n*m, R) * noise_scale
                V += np.random.randn(m*p, R) * noise_scale
                W += np.random.randn(p*n, R) * noise_scale
            
            # Run ALS
            U, V, W = alternating_least_squares(U, V, W, max_iters=100)
            
            # Check if exact
            res = compute_tensor_residuals(U, V, W)
            max_err = np.max(np.abs(res))
            
            if max_err < 1e-8:
                # Try to snap to rational values
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if check_decomposition(U_snap, V_snap, W_snap, tol=1e-9):
                    return U_snap, V_snap, W_snap
                
                U_snap = np.round(U)
                V_snap = np.round(V)
                W_snap = np.round(W)
                if check_decomposition(U_snap, V_snap, W_snap, tol=1e-9):
                    return U_snap, V_snap, W_snap
                
                # Accept if very close even without clean snap
                if max_err < 1e-10:
                    best_result = (U, V, W)
                    best_err = max_err
                    break
            else:
                if max_err < best_err:
                    best_err = max_err
                    best_result = (U, V, W)
        
        if best_result is not None and best_err < 1e-6:
            U, V, W = best_result
            # Final snap attempt
            for scale in [1, 2, 3, 4]:
                U_snap = np.round(U * scale) / scale
                V_snap = np.round(V * scale) / scale
                W_snap = np.round(W * scale) / scale
                if check_decomposition(U_snap, V_snap, W_snap, tol=1e-9):
                    return U_snap, V_snap, W_snap
            return U, V, W
        
        return None

    # Try ranks from 58 down to 45 with more restarts
    for R in range(58, 44, -1):
        if time.time() > deadline - 30:
            break
        result = try_rank(R, n_restarts=5)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
