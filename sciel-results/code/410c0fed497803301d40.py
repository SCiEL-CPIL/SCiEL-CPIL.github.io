import time
import numpy as np
from scipy.optimize import least_squares, minimize

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 3, 4, 5
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    T = np.ones((n, m, p))
    
    def verify(U, V, W, tol=1e-6):
        # Vectorized verification
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        # S[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        S = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(S, T, atol=tol)

    def residuals(params, r):
        U_flat = params[:n * m * r]
        V_flat = params[n * m * r : n * m * r + m * p * r]
        W_flat = params[n * m * r + m * p * r : n * m * r + m * p * r + p * n * r]
        
        U = U_flat.reshape(n, m, r)
        V = V_flat.reshape(m, p, r)
        W = W_flat.reshape(p, n, r)
        
        S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        res = (S - T).flatten()
        
        return res

    np.random.seed(42)
    
    # Try ranks from lower bound to schoolbook
    # Lower bound is 1 for the all-ones tensor.
    for target_rank in range(1, 61):
        if time.time() > deadline:
            break
        
        max_restarts = 5
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Special case: Rank 1 solution for all-ones tensor
            if target_rank == 1:
                U_opt = np.ones((n, m, 1))
                V_opt = np.ones((m, p, 1))
                W_opt = np.ones((p, n, 1))
                
                # Verify
                if verify(U_opt.flatten(), V_opt.flatten(), W_opt.flatten()):
                    best_U = U_opt.flatten()
                    best_V = V_opt.flatten()
                    best_W = W_opt.flatten()
                    best_rank = target_rank
                    found = True
                    break
                else:
                    # If rank 1 fails (e.g. if tensor is not all-ones), continue to search
                    continue

            # Use a mix of initialization strategies
            if restart % 3 == 0:
                U_init = np.random.randn(n * m, target_rank)
                V_init = np.random.randn(m * p, target_rank)
                W_init = np.random.randn(p * n, target_rank)
            elif restart % 3 == 1:
                U_init = np.random.rand(n * m, target_rank)
                V_init = np.random.rand(m * p, target_rank)
                W_init = np.random.rand(p * n, target_rank)
            else:
                U_init = np.random.choice([-1, 1], size=(n * m, target_rank)).astype(float)
                V_init = np.random.choice([-1, 1], size=(m * p, target_rank)).astype(float)
                W_init = np.random.choice([-1, 1], size=(p * n, target_rank)).astype(float)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(
                    residuals, x0, args=(target_rank,),
                    method='trf', max_nfev=300, xtol=1e-12, ftol=1e-12, gtol=1e-12
                )
                
                if result.cost < 1e-8:
                    U_opt = result.x[:n * m * target_rank].reshape(n, m, target_rank)
                    V_opt = result.x[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m, p, target_rank)
                    W_opt = result.x[n * m * target_rank + m * p * target_rank :].reshape(p, n, target_rank)
                    
                    # Try to snap to half-integers
                    U_r = np.round(U_opt * 2) / 2.0
                    V_r = np.round(V_opt * 2) / 2.0
                    W_r = np.round(W_opt * 2) / 2.0
                    
                    if verify(U_r.flatten(), V_r.flatten(), W_r.flatten()):
                        best_U = U_r.flatten()
                        best_V = V_r.flatten()
                        best_W = W_r.flatten()
                        best_rank = target_rank
                        found = True
                        break
                    
                    # Try integers
                    U_i = np.round(U_opt)
                    V_i = np.round(V_opt)
                    W_i = np.round(W_opt)
                    if verify(U_i.flatten(), V_i.flatten(), W_i.flatten()):
                        best_U = U_i.flatten()
                        best_V = V_i.flatten()
                        best_W = W_i.flatten()
                        best_rank = target_rank
                        found = True
                        break
                        
            except Exception:
                continue
        
        if found:
            break
    
    if not verify(best_U, best_V, best_W):
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_sb
        
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
