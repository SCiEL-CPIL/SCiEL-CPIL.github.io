import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _schoolbook(n, m, p):
    """Generate the schoolbook decomposition of rank n*m*p."""
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

def _verify_vectorized(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication using vectorized ops."""
    r = U.shape[1]
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_idx = i_idx.ravel()
    j_idx = j_idx.ravel()
    k_idx = k_idx.ravel()
    
    U_vals = U[i_idx * m + j_idx, :]
    V_vals = V[j_idx * p + k_idx, :]
    W_vals = W[k_idx * n + i_idx, :]
    
    products = U_vals * V_vals * W_vals
    sums = products.sum(axis=1)
    
    max_err = np.max(np.abs(sums - 1.0))
    return max_err < tol, max_err

def _try_round_to_half_integer(U, V, W, n, m, p):
    """Try to round the decomposition to half-integers and verify."""
    for scale in [1, 2, 4, 8, 16]:
        U_round = np.round(U * scale) / scale
        V_round = np.round(V * scale) / scale
        W_round = np.round(W * scale) / scale
        valid, _ = _verify_vectorized(U_round, V_round, W_round, n, m, p, tol=1e-9)
        if valid:
            return U_round, V_round, W_round
    return None

def _als_decompose(n, m, p, r, U_init=None, V_init=None, W_init=None, max_iter=500, tol=1e-12):
    """Alternating least squares for tensor decomposition."""
    if U_init is None:
        U = np.random.randn(n * m, r) * 0.1
    else:
        U = U_init.copy()
    if V_init is None:
        V = np.random.randn(m * p, r) * 0.1
    else:
        V = V_init.copy()
    if W_init is None:
        W = np.random.randn(p * n, r) * 0.1
    else:
        W = W_init.copy()
    
    for iteration in range(max_iter):
        W_new = np.zeros((p * n, r))
        for k in range(p):
            for i in range(n):
                coeffs = np.zeros(r)
                for j in range(m):
                    coeffs += U[i * m + j, :] * V[j * p + k, :]
                norm_sq = np.dot(coeffs, coeffs)
                if norm_sq > 1e-12:
                    W_new[k * n + i, :] = coeffs / norm_sq
        W = W_new
        
        U_new = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                coeffs = np.zeros(r)
                for k in range(p):
                    coeffs += V[j * p + k, :] * W[k * n + i, :]
                norm_sq = np.dot(coeffs, coeffs)
                if norm_sq > 1e-12:
                    U_new[i * m + j, :] = coeffs / norm_sq
        U = U_new
        
        V_new = np.zeros((m * p, r))
        for j in range(m):
            for k in range(p):
                coeffs = np.zeros(r)
                for i in range(n):
                    coeffs += U[i * m + j, :] * W[k * n + i, :]
                norm_sq = np.dot(coeffs, coeffs)
                if norm_sq > 1e-12:
                    V_new[j * p + k, :] = coeffs / norm_sq
        V = V_new
        
        if iteration % 20 == 0 or iteration == max_iter - 1:
            valid, max_err = _verify_vectorized(U, V, W, n, m, p, tol=tol)
            if valid:
                break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = rank_schoolbook
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    start_time = time.time()
    deadline = start_time + 130
    
    target_ranks = list(range(119, 99, -1))
    
    for r in target_ranks:
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(20):
            if time.time() > deadline:
                break
            
            if trial == 0:
                U = U_school[:, :r].copy()
                V = V_school[:, :r].copy()
                W = W_school[:, :r].copy()
            elif trial < 8:
                scale = 0.2 + 0.1 * trial
                U = np.random.randn(n * m, r) * scale
                V = np.random.randn(m * p, r) * scale
                W = np.random.randn(p * n, r) * scale
            else:
                U = np.random.randn(n * m, r) * 0.3
                V = np.random.randn(m * p, r) * 0.3
                W = np.random.randn(p * n, r) * 0.3
                mask_u = np.random.rand(n * m, r) < 0.3
                mask_v = np.random.rand(m * p, r) < 0.3
                mask_w = np.random.rand(p * n, r) < 0.3
                U[mask_u] = 0
                V[mask_v] = 0
                W[mask_w] = 0
            
            U_out, V_out, W_out = _als_decompose(n, m, p, r, U_init=U, V_init=V, W_init=W, max_iter=300, tol=1e-12)
            
            valid, max_err = _verify_vectorized(U_out, V_out, W_out, n, m, p, tol=1e-8)
            if valid:
                rounded = _try_round_to_half_integer(U_out, V_out, W_out, n, m, p)
                if rounded is not None:
                    U_round, V_round, W_round = rounded
                    valid_round, _ = _verify_vectorized(U_round, V_round, W_round, n, m, p, tol=1e-9)
                    if valid_round:
                        best_rank = r
                        best_U = U_round.copy()
                        best_V = V_round.copy()
                        best_W = W_round.copy()
                        found = True
                        break
        
        if found:
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
