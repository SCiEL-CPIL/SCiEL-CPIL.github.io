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

    def try_rank(R, max_starts=5, max_iter=500):
        # Variables: U (U_dim x R), V (V_dim x R), W (W_dim x R)
        # Total size: R * (U_dim + V_dim + W_dim)
        size_U = U_dim * R
        size_V = V_dim * R
        size_W = W_dim * R
        total_size = size_U + size_V + size_W
        
        def objective(x):
            U = x[:size_U].reshape(U_dim, R)
            V = x[size_U:size_U+size_V].reshape(V_dim, R)
            W = x[size_U+size_V:].reshape(W_dim, R)
            # Compute tensor product
            # prod[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            return np.sum(diff * diff)
        
        for start in range(max_starts):
            np.random.seed(1000 + start)
            # Initialize with small random values
            x0 = np.random.randn(total_size) * 0.1
            
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < 1e-6:
                x_opt = res.x
                U = x_opt[:size_U].reshape(U_dim, R)
                V = x_opt[size_U:size_U+size_V].reshape(V_dim, R)
                W = x_opt[size_U+size_V:].reshape(W_dim, R)
                
                U_s = snap_to_half(U)
                V_s = snap_to_half(V)
                W_s = snap_to_half(W)
                
                if validate(U_s, V_s, W_s):
                    return U_s, V_s, W_s
        return None

    # Try lower ranks first
    for R in [45, 44, 43, 42]:
        sol = try_rank(R, max_starts=10, max_iter=200)
        if sol:
            U, V, W = sol
            return {
                "U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]
            }
            
    # Fallback: Rank 46
    # Re-implementing the constructive rank 46 solution to ensure validity
    # This is a fallback, not the primary design
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
        
    U_s = snap_to_half(U)
    V_s = snap_to_half(V)
    W_s = snap_to_half(W)
    
    return {
        "U": [[float(x) for x in row] for row in U_s],
        "V": [[float(x) for x in row] for row in V_s],
        "W": [[float(x) for x in row] for row in W_s]
    }
