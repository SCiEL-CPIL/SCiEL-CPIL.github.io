"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition search."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]
_DEADLINE = 140.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed()

def construct():
    n, m, p = DIMENSIONS
    # Tensor T has shape (n*m, m*p, p*n) with T[i*m+j, j*p+k, k*n+i] = 1
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
    
    def _flatten_params(U, V, W, r):
        return np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    
    def _unflatten(params, r):
        sz_u = n*m*r
        sz_v = m*p*r
        sz_w = p*n*r
        U = params[:sz_u].reshape(n*m, r)
        V = params[sz_u:sz_u+sz_v].reshape(m*p, r)
        W = params[sz_u+sz_v:sz_u+sz_v+sz_w].reshape(p*n, r)
        return U, V, W
    
    def _objective(params, r):
        U, V, W = _unflatten(params, r)
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx.reshape(-1) - T_flat
        return np.sum(diff*diff)
    
    def _objective_grad(params, r):
        U, V, W = _unflatten(params, r)
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx - T
        gU = np.einsum('abc,br,cr->ar', diff, V, W)
        gV = np.einsum('abc,ar,cr->br', diff, U, W)
        gW = np.einsum('abc,ar,br->cr', diff, U, V)
        return np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])
    
    def _verify_exact(U, V, W, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        if np.max(np.abs(T_approx - T)) > tol:
            return False
        for mat in [U, V, W]:
            for val in mat:
                nearest = round(val * 2) / 2
                if abs(val - nearest) > 1e-6:
                    return False
        return True
    
    def _round_to_half(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r

    rng = np.random.default_rng(42)
    
    # Focus on rank 14 and 13, as 18 is the fallback.
    # Rank 14 is the known minimum for 2x3x3.
    target_ranks = [14, 13]
    
    for rank in target_ranks:
        if _remaining() < 20:
            break
        
        found = False
        max_restarts = 50
        
        for restart in range(max_restarts):
            if _remaining() < 10:
                break
            
            # Structured initialization based on Strassen-like blocks
            if restart % 2 == 0:
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                
                # Initialize first 7 columns with Strassen pattern for 2x2x2 subblock (i=0,1; j=0,1; k=0,1)
                # Map (i,j,k) to indices
                # Strassen M1: (A00+A11)(B00+B11)
                # U: A00, A11 ; V: B00, B11 ; W: C00, C11? No, W is output.
                # This is tricky to map directly without knowing the exact decomposition.
                # Instead, use random sparse initialization which often works better for tensor rank.
                U = rng.standard_normal((n*m, rank)) * 0.5
                V = rng.standard_normal((m*p, rank)) * 0.5
                W = rng.standard_normal((p*n, rank)) * 0.5
            else:
                # Sparse half-integer initialization
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for r in range(rank):
                    idx_u = rng.choice(n*m, size=2, replace=False)
                    idx_v = rng.choice(m*p, size=2, replace=False)
                    idx_w = rng.choice(p*n, size=2, replace=False)
                    U[idx_u, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
                    V[idx_v, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
                    W[idx_w, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
            
            params = _flatten_params(U, V, W, rank)
            
            try:
                res = minimize(
                    _objective, params, args=(rank,),
                    jac=_objective_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-10}
                )
                
                if res.fun < 1e-8:
                    U_opt, V_opt, W_opt = _unflatten(res.x, rank)
                    U_r, V_r, W_r = _round_to_half(U_opt, V_opt, W_opt)
                    
                    if _verify_exact(U_r, V_r, W_r, tol=1e-7):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = rank
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            continue

    if best_rank < rank_sb:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        U_v = np.array(U_list)
        V_v = np.array(V_list)
        W_v = np.array(W_list)
        T_approx = np.einsum('ar,br,cr->abc', U_v, V_v, W_v)
        if np.max(np.abs(T_approx - T)) < 1e-7:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    U_list = [[int(x) for x in row] for row in U0]
    V_list = [[int(x) for x in row] for row in V0]
    W_list = [[int(x) for x in row] for row in W0]
    return {"U": U_list, "V": V_list, "W": W_list}
