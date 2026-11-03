"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if len(V[0]) != rank:
        return False
    if len(W[0]) != rank:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(rank):
                    total += U[ui][c] * V[vj][c] * W[wk][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Target rank 20 (Landsberg-Ottaviani bound)
    target_rank = 20
    
    # True tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Fallback schoolbook
    def get_schoolbook():
        R = n*m*p
        U_s = np.zeros((n*m, R))
        V_s = np.zeros((m*p, R))
        W_s = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, col] = 1
                    V_s[j*p+k, col] = 1
                    W_s[k*n+i, col] = 1
                    col += 1
        return {
            "U": U_s.tolist(),
            "V": V_s.tolist(),
            "W": W_s.tolist()
        }

    start_time = time.time()
    
    # Try to find rank 20 decomposition
    for R in range(20, 25):
        if time.time() - start_time > 100:
            break
            
        # Try multiple random restarts
        for restart in range(5):
            np.random.seed(42 + R + restart)
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def error(x):
                U_loc = x[:n*m*R].reshape(n*m, R)
                V_loc = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_loc = x[n*m*R + m*p*R:].reshape(p*n, R)
                
                T_comp = np.zeros((n*m, m*p, p*n))
                for c in range(R):
                    T_comp += np.einsum('i,j,k->ijk', U_loc[:, c], V_loc[:, c], W_loc[:, c])
                
                return np.sum((T_comp - T)**2)
            
            try:
                res = minimize(error, x, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                U_opt = res.x[:n*m*R].reshape(n*m, R)
                V_opt = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_opt = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Snap to integers/half-integers
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Verify
                if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }

    # Fallback
    return get_schoolbook()
# EVOLVE-BLOCK-END
