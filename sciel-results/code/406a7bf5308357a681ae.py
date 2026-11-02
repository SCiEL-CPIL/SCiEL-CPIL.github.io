"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = 10
    
    # Define the target tensor T of shape (4, 6, 6)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = i * p + k
                T[a, b, c] = 1.0
                
    def compute_error(params):
        U_flat, V_flat, W_flat = params
        U = U_flat.reshape(n*m, target_rank)
        V = V_flat.reshape(m*p, target_rank)
        W = W_flat.reshape(p*n, target_rank)
        
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(target_rank):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
        return np.sum((T_recon - T)**2)

    def compute_params(params):
        U_flat, V_flat, W_flat = params
        U = U_flat.reshape(n*m, target_rank)
        V = V_flat.reshape(m*p, target_rank)
        W = W_flat.reshape(p*n, target_rank)
        return U, V, W

    best_error = float('inf')
    best_params = None
    start_time = time.time()
    time_limit = 110 # Safety margin
    
    # Rank 11 solution for warm start
    U11 = np.array([
        [1, 1, 1, 1, 1, -1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1, 0, 0, 0, 1]
    ])
    V11 = np.array([
        [1, 1, 0, -1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
    ])
    W11 = np.array([
        [1, 0, 0, 1, -1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    ])
    
    init1 = np.concatenate([U11[:, :10].flatten(), V11[:, :10].flatten(), W11[:, :10].flatten()])
    
    np.random.seed(42)
    initial_guesses = [init1]
    # Add random restarts
    for _ in range(10):
        initial_guesses.append(np.random.normal(size=4*10 + 6*10 + 6*10))
        
    for i, x0 in enumerate(initial_guesses):
        if time.time() - start_time > time_limit:
            break
        try:
            res = minimize(compute_error, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
        except Exception:
            pass
            
    if best_params is not None and best_error < 1e-6:
        U, V, W = compute_params(best_params)
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(target_rank):
            T_recon += np.einsum('a,b,c->abc', U_snap[:, r], V_snap[:, r], W_snap[:, r])
            
        if np.allclose(T_recon, T, atol=1e-9):
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
    # Fallback to Rank 11
    U = [[0] * 11 for _ in range(4)]
    V = [[0] * 11 for _ in range(6)]
    W = [[0] * 11 for _ in range(6)]
    
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[4][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1; W[3][1] += -1
    
    U[0][2] += 1
    V[1][2] += 1; V[4][2] += -1
    W[2][2] += 1; W[3][2] += 1
    
    U[3][3] += 1
    V[3][3] += 1; V[0][3] += -1
    W[0][3] += 1; W[1][3] += 1
    
    U[0][4] += 1; U[1][4] += 1
    V[4][4] += 1
    W[0][4] += -1; W[2][4] += 1
    
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    U[1][6] += 1; U[3][6] += -1
    V[3][6] += 1; V[4][6] += 1
    W[0][6] += 1
    
    U[0][7] += 1
    V[2][7] += 1
    W[4][7] += 1
    
    U[1][8] += 1
    V[5][8] += 1
    W[4][8] += 1
    
    U[2][9] += 1
    V[2][9] += 1
    W[5][9] += 1
    
    U[3][10] += 1
    V[5][10] += 1
    W[5][10] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
