"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

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
    # We seek rank-r decomposition: T = sum_c U[:,c] * V[:,c] * W[:,c]
    # i.e., T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
    # Build the target tensor
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
        """Check if the decomposition is exact and entries are integer/half-integer"""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        if np.max(np.abs(T_approx - T)) > tol:
            return False
        # Check entries are close to integer or half-integer
        for mat in [U, V, W]:
            for val in mat:
                nearest = round(val * 2) / 2
                if abs(val - nearest) > 1e-6:
                    return False
        return True
    
    def _round_to_half(U, V, W):
        """Round entries to nearest half-integer"""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r
    
    def _polish(U, V, W, r, max_iter=50):
        """High-precision polish pass using alternating least squares"""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iter):
            if _remaining() < 5:
                break
            
            # Update U: for each row a, solve least squares
            for a in range(n*m):
                # T[a, :, :] = U[a, :] * V.T * W.T (broadcast)
                # Reshape: target = T[a, :, :].reshape(-1)
                # Design matrix: [V.T * W.T] element-wise, reshaped
                VW = (V.T * W.T).reshape(m*p*p*n, r)
                target = T[a, :, :].reshape(-1)
                if np.linalg.matrix_rank(VW) > 0:
                    sol, residuals, rank_v, sv = np.linalg.lstsq(VW, target, rcond=None)
                    U[a, :] = sol
            
            # Update V
            for b in range(m*p):
                UW = (U.T * W.T).reshape(n*m*p*n, r)
                target = T[:, b, :].reshape(-1)
                if np.linalg.matrix_rank(UW) > 0:
                    sol, residuals, rank_v, sv = np.linalg.lstsq(UW, target, rcond=None)
                    V[b, :] = sol
            
            # Update W
            for c in range(p*n):
                UV = (U.T * V.T).reshape(n*m*m*p, r)
                target = T[:, :, c].reshape(-1)
                if np.linalg.matrix_rank(UV) > 0:
                    sol, residuals, rank_v, sv = np.linalg.lstsq(UV, target, rcond=None)
                    W[c, :] = sol
        
        return U, V, W
    
    # Try ranks from 17 down to 13
    # Known lower bound for 2x3x3 is around 13
    rng = np.random.default_rng(42)
    
    # Strategy: For each rank, try multiple initialization strategies
    # 1. Random Gaussian
    # 2. Structured based on tensor symmetry  
    # 3. Sparse random with half-integer values
    # 4. Perturbed schoolbook (remove random columns)
    
    max_restarts_per_rank = 40
    
    for rank in range(rank_sb - 1, 12, -1):
        if _remaining() < 15:
            break
        
        found = False
        for restart in range(max_restarts_per_rank):
            if _remaining() < 10:
                break
            
            # Choose initialization strategy based on restart number
            strategy = restart % 4
            
            if strategy == 0:
                # Random Gaussian
                U = rng.standard_normal((n*m, rank)) * 0.5
                V = rng.standard_normal((m*p, rank)) * 0.5
                W = rng.standard_normal((p*n, rank)) * 0.5
            elif strategy == 1:
                # Structured: try to exploit the tensor's algebraic structure
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for r in range(rank):
                    n_nonzero_u = max(1, int(n*m * 0.3))
                    n_nonzero_v = max(1, int(m*p * 0.3))
                    n_nonzero_w = max(1, int(p*n * 0.3))
                    
                    idx_u = rng.choice(n*m, size=min(n_nonzero_u, n*m), replace=False)
                    idx_v = rng.choice(m*p, size=min(n_nonzero_v, m*p), replace=False)
                    idx_w = rng.choice(p*n, size=min(n_nonzero_w, p*n), replace=False)
                    
                    U[idx_u, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
                    V[idx_v, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
                    W[idx_w, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
            elif strategy == 2:
                # Sparse random with half-integer values
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for r in range(rank):
                    n_nonzero_u = max(1, int(n*m * 0.25))
                    n_nonzero_v = max(1, int(m*p * 0.25))
                    n_nonzero_w = max(1, int(p*n * 0.25))
                    
                    idx_u = rng.choice(n*m, size=min(n_nonzero_u, n*m), replace=False)
                    idx_v = rng.choice(m*p, size=min(n_nonzero_v, m*p), replace=False)
                    idx_w = rng.choice(p*n, size=min(n_nonzero_w, p*n), replace=False)
                    
                    U[idx_u, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
                    V[idx_v, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
                    W[idx_w, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
            else:
                # Perturbed schoolbook: start from schoolbook and remove columns
                # Keep 'rank' columns from schoolbook, perturb slightly
                if rank >= rank_sb:
                    U = U0.copy()
                    V = V0.copy()
                    W = W0.copy()
                else:
                    # Randomly select 'rank' columns from schoolbook
                    cols = rng.choice(rank_sb, size=rank, replace=False)
                    U = U0[:, cols] + rng.standard_normal((n*m, rank)) * 0.1
                    V = V0[:, cols] + rng.standard_normal((m*p, rank)) * 0.1
                    W = W0[:, cols] + rng.standard_normal((p*n, rank)) * 0.1
            
            params = _flatten_params(U, V, W, rank)
            
            try:
                # Try L-BFGS-B optimization
                best_res = None
                best_obj = np.inf
                
                res = minimize(
                    _objective, params, args=(rank,),
                    jac=_objective_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12}
                )
                best_obj = res.fun
                best_res = res
                
                if best_obj < 1e-6:
                    U_opt, V_opt, W_opt = _unflatten(best_res.x, rank)
                    
                    # Polish pass
                    U_pol, V_pol, W_pol = _polish(U_opt, V_opt, W_opt, rank, max_iter=30)
                    
                    # Round to half-integers
                    U_r, V_r, W_r = _round_to_half(U_pol, V_pol, W_pol)
                    
                    if _verify_exact(U_r, V_r, W_r, tol=1e-7):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = rank
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            continue
    
    # If we found a better rank, return it
    if best_rank < rank_sb:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        # Final verification
        U_v = np.array(U_list)
        V_v = np.array(V_list)
        W_v = np.array(W_list)
        T_approx = np.einsum('ar,br,cr->abc', U_v, V_v, W_v)
        if np.max(np.abs(T_approx - T)) < 1e-7:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U_list = [[int(x) for x in row] for row in U0]
    V_list = [[int(x) for x in row] for row in V0]
    W_list = [[int(x) for x in row] for row in W0]
    return {"U": U_list, "V": V_list, "W": W_list}
