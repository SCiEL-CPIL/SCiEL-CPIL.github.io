import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=5
    # Tensor T has shape (n*m, m*p, p*n) = (12, 20, 15)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook as fallback
    best_rank = n * m * p
    U_s = np.zeros((n*m, best_rank))
    V_s = np.zeros((m*p, best_rank))
    W_s = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1.0
                V_s[j*p+k, col] = 1.0
                W_s[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = None, None, None
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    # Define residual function for least_squares
    def residual(x, rank):
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return (T - T_recon).flatten()
    
    # Try ranks from 56 down to 20 (56 is known best)
    # We prioritize 56, 55, etc.
    ranks_to_try = [56, 55, 54, 53, 52, 51, 50]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try multiple initializations
        found = False
        for init_seed in range(10):
            if time.time() > deadline:
                break
                
            np.random.seed(init_seed)
            # Initialize with random vectors, scaled
            U0 = np.random.randn(n*m, rank)
            V0 = np.random.randn(m*p, rank)
            W0 = np.random.randn(p*n, rank)
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                res = least_squares(residual, x0, args=(rank,), method='trf', max_nfev=5000)
                err = np.linalg.norm(res.fun)
                
                if err < 1e-6:
                    U_opt = res.x[:n*m*rank].reshape(n*m, rank)
                    V_opt = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_opt = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Try to snap to half-integers
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    T_recon = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                    if np.allclose(T_recon, T, atol=1e-9):
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        best_rank = rank
                        found = True
                        break
                    
                    # Try integer snap
                    U_snap = np.round(U_opt)
                    V_snap = np.round(V_opt)
                    W_snap = np.round(W_opt)
                    T_recon = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                    if np.allclose(T_recon, T, atol=1e-9):
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        best_rank = rank
                        found = True
                        break
                        
            except Exception:
                continue
                
        if found:
            break
            
    if best_U is None:
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
