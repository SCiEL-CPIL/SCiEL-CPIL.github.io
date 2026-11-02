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
    nm, mp, pn = n * m, m * p, p * n
    n_residuals = n * m * p

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

    def verify_solution(U_sol, V_sol, W_sol, rank):
        """Verify that the solution exactly decomposes the matrix multiplication tensor."""
        U_r = U_sol.reshape(n, m, rank)
        V_r = V_sol.reshape(m, p, rank)
        W_r = W_sol.reshape(p, n, rank)
        T_approx = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.max(np.abs(T_approx - 1.0)) < 1e-10

    def polish_solution(U_init, V_init, W_init, rank):
        """High-precision polish pass to achieve exact decomposition."""
        x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
        
        # Use tighter tolerance and more iterations
        try:
            res = least_squares(residual_func, x0, args=(rank,), method='trf', 
                               max_nfev=5000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
        except Exception:
            return None
            
        if res.cost < 1e-12:
            U_sol = res.x[:rank*nm].reshape(nm, rank)
            V_sol = res.x[rank*nm:rank*(nm+mp)].reshape(mp, rank)
            W_sol = res.x[rank*(nm+mp):].reshape(pn, rank)
            
            if verify_solution(U_sol, V_sol, W_sol, rank):
                return U_sol, V_sol, W_sol
        
        return None

    for rank in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
            
        found = False
        # More restarts for smaller ranks
        max_restarts = 30 if rank < 30 else 15
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Multiple initialization strategies
            init_strategy = restart % 3
            
            if init_strategy == 0:
                # Pure random
                scale = 1.0
                U0 = np.random.randn(n*m, rank) * scale
                V0 = np.random.randn(m*p, rank) * scale
                W0 = np.random.randn(p*n, rank) * scale
            elif init_strategy == 1:
                # Structured initialization based on schoolbook patterns
                U0 = np.zeros((n*m, rank))
                V0 = np.zeros((m*p, rank))
                W0 = np.zeros((p*n, rank))
                for c in range(rank):
                    for i in range(n):
                        for j in range(m):
                            U0[i*m+j, c] = np.random.choice([-1, 0, 1])
                    for j in range(m):
                        for k in range(p):
                            V0[j*p+k, c] = np.random.choice([-1, 0, 1])
                    for k in range(p):
                        for i in range(n):
                            W0[k*n+i, c] = np.random.choice([-1, 0, 1])
            else:
                # Low-rank approximation initialization
                U0 = np.random.randn(n*m, min(rank, 8)) * 0.5
                V0 = np.random.randn(m*p, min(rank, 8)) * 0.5
                W0 = np.random.randn(p*n, min(rank, 8)) * 0.5
                # Pad with zeros if needed
                if rank > 8:
                    U0 = np.hstack([U0, np.zeros((n*m, rank-8))])
                    V0 = np.hstack([V0, np.zeros((m*p, rank-8))])
                    W0 = np.hstack([W0, np.zeros((p*n, rank-8))])
            
            x0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
            
            try:
                # Use appropriate method based on problem dimensions
                n_params = rank * (nm + mp + pn)
                method = 'lm' if n_params >= n_residuals else 'trf'
                
                res = least_squares(residual_func, x0, args=(rank,), method=method, max_nfev=3000)
            except Exception:
                continue
                
            if res.cost < 1e-8:
                # Verify
                U_sol = res.x[:rank*nm].reshape(nm, rank)
                V_sol = res.x[rank*nm:rank*(nm+mp)].reshape(mp, rank)
                W_sol = res.x[rank*(nm+mp):].reshape(pn, rank)
                
                if verify_solution(U_sol, V_sol, W_sol, rank):
                    # Try to polish for even better precision
                    polished = polish_solution(U_sol, V_sol, W_sol, rank)
                    if polished is not None:
                        U_sol, V_sol, W_sol = polished
                    
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    best_rank = rank
                    found = True
                    break
                    
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
