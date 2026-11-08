import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-5):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_to_half(arr):
        return np.round(arr * 2) / 2

    def objective(params, r):
        U = params[:r * U_dim].reshape(r, U_dim).T
        V = params[r * U_dim : r * (U_dim + V_dim)].reshape(r, V_dim).T
        W = params[r * (U_dim + V_dim):].reshape(r, W_dim).T
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    def try_rank(r, num_restarts=20, time_limit_per_restart=2.0):
        np.random.seed(42 + r)
        for _ in range(num_restarts):
            start_time = time.time()
            # Initialization: Random
            U = np.random.randn(U_dim, r) * 0.5
            V = np.random.randn(V_dim, r) * 0.5
            W = np.random.randn(W_dim, r) * 0.5
            
            # Flatten
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Optimization
            res = minimize(
                objective, 
                params, 
                args=(r,), 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() - start_time > time_limit_per_restart:
                continue
                
            if res.fun < 1e-4:
                # Extract
                U_opt = res.x[:r * U_dim].reshape(r, U_dim).T
                V_opt = res.x[r * U_dim : r * (U_dim + V_dim)].reshape(r, V_dim).T
                W_opt = res.x[r * (U_dim + V_dim):].reshape(r, W_dim).T
                
                # Snap
                U_snap = snap_to_half(U_opt)
                V_snap = snap_to_half(V_opt)
                W_snap = snap_to_half(W_opt)
                
                if validate(U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap
            elif res.fun < 1e-2:
                # Sometimes close enough to snap if we do a second pass or just check
                U_opt = res.x[:r * U_dim].reshape(r, U_dim).T
                V_opt = res.x[r * U_dim : r * (U_dim + V_dim)].reshape(r, V_dim).T
                W_opt = res.x[r * (U_dim + V_dim):].reshape(r, W_dim).T
                
                U_snap = snap_to_half(U_opt)
                V_snap = snap_to_half(V_opt)
                W_snap = snap_to_half(W_opt)
                
                if validate(U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap

        return None

    # Try to find a rank lower than 46
    # We know 46 is possible. Try 45, 44, 43...
    for r in range(45, 30, -1):
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            return {
                "U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]
            }

    # Fallback: Return a valid rank 46 decomposition
    # Re-implementing the known rank 46 solution from the current program
    rank = 46
    U = np.zeros((U_dim, rank))
    V = np.zeros((V_dim, rank))
    W = np.zeros((W_dim, rank))
    
    def add_strassen(j_off, k_off, col_start):
        c = col_start
        U[0 * m + j_off][c] += 1
        U[1 * m + (j_off + 1)][c] += 1
        V[j_off * p + k_off][c] += 1
        V[(j_off + 1) * p + (k_off + 1)][c] += 1
        W[k_off * n + 0][c] += 1
        W[(k_off + 1) * n + 1][c] += 1
        c += 1
        U[1 * m + j_off][c] += 1
        U[1 * m + (j_off + 1)][c] += 1
        V[j_off * p + k_off][c] += 1
        W[k_off * n + 1][c] += 1
        W[(k_off + 1) * n + 1][c] -= 1
        c += 1
        U[0 * m + j_off][c] += 1
        V[j_off * p + (k_off + 1)][c] += 1
        V[(j_off + 1) * p + (k_off + 1)][c] -= 1
        W[(k_off + 1) * n + 0][c] += 1
        W[(k_off + 1) * n + 1][c] += 1
        c += 1
        U[1 * m + (j_off + 1)][c] += 1
        V[(j_off + 1) * p + k_off][c] += 1
        V[j_off * p + k_off][c] -= 1
        W[k_off * n + 0][c] += 1
        W[k_off * n + 1][c] += 1
        c += 1
        U[0 * m + j_off][c] += 1
        U[0 * m + (j_off + 1)][c] += 1
        V[(j_off + 1) * p + (k_off + 1)][c] += 1
        W[k_off * n + 0][c] -= 1
        W[(k_off + 1) * n + 0][c] += 1
        c += 1
        U[1 * m + j_off][c] += 1
        U[0 * m + j_off][c] -= 1
        V[j_off * p + k_off][c] += 1
        V[j_off * p + (k_off + 1)][c] += 1
        W[(k_off + 1) * n + 1][c] += 1
        c += 1
        U[0 * m + (j_off + 1)][c] += 1
        U[1 * m + (j_off + 1)][c] -= 1
        V[(j_off + 1) * p + k_off][c] += 1
        V[(j_off + 1) * p + (k_off + 1)][c] += 1
        W[k_off * n + 0][c] += 1

    add_strassen(0, 0, 0)
    add_strassen(0, 2, 7)
    add_strassen(2, 0, 14)
    add_strassen(2, 2, 21)
    
    col = 28
    for i in range(2):
        for k in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    for i in range(2):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    for i in range(2):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1
        
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
