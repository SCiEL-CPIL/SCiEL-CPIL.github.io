import numpy as np
from scipy.optimize import least_squares
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = 115.0 # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    # Target tensor T[i, j, j_prime, k] = delta_{j, j_prime}
    # Shape: (n, m, m, p)
    T = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k] = 1.0
                
    def compute_M(U, V, W, R):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        M = np.zeros((n, m, m, p))
        for r in range(R):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            
            term = np.einsum('ki,ij,ak->ijak', w_r, u_r, v_r)
            M += term
        return M
        
    def residuals(params, R):
        size_U = nm * R
        size_V = mp * R
        U = params[:size_U].reshape(nm, R)
        V = params[size_U:size_U + size_V].reshape(mp, R)
        W = params[size_U + size_V:].reshape(pn, R)
        
        M = compute_M(U, V, W, R)
        return (M - T).flatten()
        
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Focus on Rank 11
    target_rank = 11
    
    # Multiple random restarts
    num_restarts = 20
    rng = np.random.default_rng(42)
    
    for r_idx in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Scale matters. Let's try small random values.
        U0 = rng.normal(0, 1, (nm, target_rank))
        V0 = rng.normal(0, 1, (mp, target_rank))
        W0 = rng.normal(0, 1, (pn, target_rank))
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(residuals, params0, args=(target_rank,), method='lm', max_nfev=500)
            
            if result.success or result.cost < 1e-6:
                U_opt = result.x[:nm * target_rank].reshape(nm, target_rank)
                V_opt = result.x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                W_opt = result.x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)
                
                M_opt = compute_M(U_opt, V_opt, W_opt, target_rank)
                err = np.linalg.norm(M_opt - T)
                
                if err < 1e-5:
                    # Snap to half-integers
                    U_snap = np.round(U_opt * 2) / 2.0
                    V_snap = np.round(V_opt * 2) / 2.0
                    W_snap = np.round(W_opt * 2) / 2.0
                    
                    M_snap = compute_M(U_snap, V_snap, W_snap, target_rank)
                    if np.allclose(M_snap, T, atol=1e-9):
                        best_rank = target_rank
                        best_U = U_snap
                        best_V = V_snap
                        best_W = W_snap
                        break # Found it!
                        
        except Exception:
            pass

    # Convert best solution to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
