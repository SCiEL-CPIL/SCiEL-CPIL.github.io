"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]
_DEADLINE = 140.0
_START_TIME = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    # Tensor T: T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    # We seek rank-r: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_flat = T.reshape(-1)
    
    # Schoolbook rank = 18
    rank_sb = n * m * p
    U0 = np.zeros((n*m, rank_sb))
    V0 = np.zeros((m*p, rank_sb))
    W0 = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_sb
    
    def _objective(params, r):
        sz_u = n*m*r
        sz_v = m*p*r
        U = params[:sz_u].reshape(n*m, r)
        V = params[sz_u:sz_u+sz_v].reshape(m*p, r)
        W = params[sz_u+sz_v:].reshape(p*n, r)
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx.reshape(-1) - T_flat
        return np.sum(diff*diff)
    
    def _objective_grad(params, r):
        sz_u = n*m*r
        sz_v = m*p*r
        U = params[:sz_u].reshape(n*m, r)
        V = params[sz_u:sz_u+sz_v].reshape(m*p, r)
        W = params[sz_u+sz_v:].reshape(p*n, r)
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx - T
        gU = np.einsum('abc,br,cr->ar', diff, V, W)
        gV = np.einsum('abc,ar,cr->br', diff, U, W)
        gW = np.einsum('abc,ar,br->cr', diff, U, V)
        return np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])
    
    rng = np.random.default_rng(42)
    
    # Try ranks from 17 down to 13
    for rank in range(rank_sb - 1, 12, -1):
        if _remaining() < 15:
            break
        found = False
        max_restarts = 30
        for restart in range(max_restarts):
            if _remaining() < 5:
                break
            # Random initialization
            U = rng.standard_normal((n*m, rank)) * 0.5
            V = rng.standard_normal((m*p, rank)) * 0.5
            W = rng.standard_normal((p*n, rank)) * 0.5
            params = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            
            try:
                res = minimize(
                    _objective, params, args=(rank,),
                    jac=lambda p, r: _objective_grad(p, r),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'gtol': 1e-12, 'ftol': 1e-14, 'maxfun': 5000}
                )
                if _remaining() < 5:
                    break
                if res.fun < 1e-12:
                    sz_u = n*m*rank
                    sz_v = m*p*rank
                    U_best = res.x[:sz_u].reshape(n*m, rank)
                    V_best = res.x[sz_u:sz_u+sz_v].reshape(m*p, rank)
                    W_best = res.x[sz_u+sz_v:].reshape(p*n, rank)
                    best_U, best_V, best_W = U_best, V_best, W_best
                    best_rank = rank
                    found = True
                    break
            except Exception:
                pass
        
        if found:
            continue
        else:
            # If we can't find at this rank, lower ranks are even harder
            # But keep trying a few more ranks
            pass
    
    # Convert to lists with rounding to nearest half-integer for exactness
    if best_rank < rank_sb:
        U_list = [[round(x*2)/2 for x in row] for row in best_U]
        V_list = [[round(x*2)/2 for x in row] for row in best_V]
        W_list = [[round(x*2)/2 for x in row] for row in best_W]
        
        # Verify the rounded version
        U_v = np.array(U_list, dtype=float)
        V_v = np.array(V_list, dtype=float)
        W_v = np.array(W_list, dtype=float)
        T_approx = np.einsum('ar,br,cr->abc', U_v, V_v, W_v)
        if np.max(np.abs(T_approx - T)) < 1e-9:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U_list = [[int(x) for x in row] for row in U0]
    V_list = [[int(x) for x in row] for row in V0]
    W_list = [[int(x) for x in row] for row in W0]
    return {"U": U_list, "V": V_list, "W": W_list}
