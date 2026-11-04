import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    R = np.einsum('ar,br,cr->abc', U, V, W)
    return np.linalg.norm(R - T) < tol

def _check_half_integer(M, tol=1e-9):
    vals = M.flatten()
    return np.all(np.isclose(2 * vals, np.round(2 * vals), atol=tol))

def construct():
    start_time = time.time()
    time_limit = 145
    
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    # Build the linear system matrix for W given U,V
    # A[(a,b,c), (c,r)] = U[a,r]*V[b,r]
    # This depends on U and V, so we rebuild each iteration
    
    def solve_for_W(U, V, r):
        A = np.zeros((n*m * m*p * p*n, p*n * r))
        t = np.zeros(n*m * m*p * p*n)
        
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    eq_idx = a * (m*p) * (p*n) + b * (p*n) + c
                    t[eq_idx] = T[a, b, c]
                    for r_idx in range(r):
                        A[eq_idx, c * r + r_idx] = U[a, r_idx] * V[b, r_idx]
        
        w_flat, _, _, _ = np.linalg.lstsq(A, t, rcond=None)
        return w_flat.reshape(p*n, r)
    
    def compute_error(U, V, W):
        R = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(R - T)
    
    # Try to find rank 19 solution
    r = 19
    
    # Gradient descent on U and V
    np.random.seed(42)
    U = np.random.randn(n*m, r) * 0.5
    V = np.random.randn(m*p, r) * 0.5
    
    lr = 0.001
    for iteration in range(2000):
        if time.time() - start_time > time_limit - 10:
            break
        
        W = solve_for_W(U, V, r)
        R = np.einsum('ar,br,cr->abc', U, V, W)
        err = R - T
        err_norm = np.linalg.norm(err)
        
        if err_norm < 1e-10:
            break
        
        # Gradient of ||err||^2 with respect to U
        # d/dU[a,r] = 2 * sum_{b,c} err[a,b,c] * V[b,r] * W[c,r]
        gU = 2 * np.einsum('abc,br,cr->ar', err, V, W)
        gV = 2 * np.einsum('abc,ar,cr->br', err, U, W)
        
        U -= lr * gU
        V -= lr * gV
    
    # Check result
    W = solve_for_W(U, V, r)
    R = np.einsum('ar,br,cr->abc', U, V, W)
    err_norm = np.linalg.norm(R - T)
    
    if err_norm < 1e-8:
        U_s = np.round(2 * U) / 2
        V_s = np.round(2 * V) / 2
        W_s = np.round(2 * W) / 2
        R_s = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
        if np.linalg.norm(R_s - T) < 1e-8:
            if _check_half_integer(U_s) and _check_half_integer(V_s) and _check_half_integer(W_s):
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Fall back to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
