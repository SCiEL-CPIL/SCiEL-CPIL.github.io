"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook rank as fallback
    def schoolbook():
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

    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    # Try ranks from a lower bound upward
    # Lower bound for M_{n,m,p} rank is often cited as max(nm, mp, pn) is FALSE.
    # Actually, rank >= max(nm, mp, pn) is a common bound for *border* rank? No.
    # Let's just start from 20.
    
    min_rank = 20
    max_rank = 60
    
    np.random.seed(42)
    
    def residual_func(params, rank):
        # params is a flat vector of length rank * (nm + mp + pn)
        nm, mp, pn = n*m, m*p, p*n
        # Reshape params to (nm, rank), (mp, rank), (pn, rank)
        U = params[:rank*nm].reshape(nm, rank)
        V = params[rank*nm:rank*(nm+mp)].reshape(mp, rank)
        W = params[rank*(nm+mp):].reshape(pn, rank)
        
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        T_approx = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        # Target is 1 everywhere
        R = T_approx - 1.0
        return R.ravel()

    for rank in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
            
        found = False
        # More restarts for smaller ranks
        max_restarts = 20 if rank < 30 else 10
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale is important
            scale = 1.0
            U0 = np.random.randn(n*m, rank) * scale
            V0 = np.random.randn(m*p, rank) * scale
            W0 = np.random.randn(p*n, rank) * scale
            
            x0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
            
            try:
                # Use Levenberg-Marquardt for underdetermined systems
                # If rank is very small such that params < residuals, use 'trf'
                nm, mp, pn = n*m, m*p, p*n
                n_params = rank * (nm + mp + pn)
                n_residuals = n * m * p
                method = 'lm' if n_params >= n_residuals else 'trf'
                
                res = least_squares(residual_func, x0, args=(rank,), method=method, max_nfev=2000)
            except Exception:
                continue
                
            if res.cost < 1e-10:
                # Verify
                nm, mp, pn = n*m, m*p, p*n
                U_sol = res.x[:rank*nm].reshape(nm, rank)
                V_sol = res.x[rank*nm:rank*(nm+mp)].reshape(mp, rank)
                W_sol = res.x[rank*(nm+mp):].reshape(pn, rank)
                
                U_r = U_sol.reshape(n, m, rank)
                V_r = V_sol.reshape(m, p, rank)
                W_r = W_sol.reshape(p, n, rank)
                T_approx = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                
                if np.max(np.abs(T_approx - 1.0)) < 1e-8:
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    best_rank = rank
                    found = True
                    break
                    
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
