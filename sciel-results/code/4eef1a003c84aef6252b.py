"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize, least_squares

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
    
    def _residual(params, r):
        U, V, W = _unflatten(params, r)
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_approx - T).reshape(-1)

    def _residual_jac(params, r):
        U, V, W = _unflatten(params, r)
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx - T
        
        # Jacobian of residual with respect to U, V, W
        # d(T_approx)/dU = V * W (outer product per column)
        # This is a 3D tensor of shape (n*m, m*p, p*n) x (n*m, r) -> (n*m*m*p*p*n, n*m*r)
        # It's easier to compute the gradient of the sum of squares and let least_squares handle it,
        # or provide the Jacobian of the residual directly.
        # scipy.optimize.least_squares can accept a callable for the Jacobian.
        
        # Let's just let it estimate the Jacobian if we don't provide it, or provide a simplified version.
        # Given the size is small (18*9*6 = 972 residuals, ~18*17 + 9*17 + 6*17 = 561 params),
        # finite difference Jacobian is acceptable but slow.
        # Let's implement the analytic Jacobian for the residual.
        
        # Residual is R = T_approx - T.
        # R_abc = sum_r U_ar V_br W_cr - T_abc
        # dR_abc / dU_ar' = delta_aa' delta_rr' V_br W_cr
        # This is sparse.
        
        # However, `least_squares` with `method='trf'` and `diff_step` might be sufficient.
        # Let's try providing the Jacobian via a helper that constructs the dense matrix.
        
        sz_u = n*m*r
        sz_v = m*p*r
        sz_w = p*n*r
        total_params = sz_u + sz_v + sz_w
        total_residuals = n*m*m*p*p*n
        
        J = np.zeros((total_residuals, total_params))
        
        # dR / dU
        # R_abc = sum_r U_ar V_br W_cr
        # dR_abc / dU_ar = V_br W_cr
        # Indexing:
        # Residual index: a, b, c -> idx = (a * m * p + b) * p * n + c ? No.
        # T shape: (n*m, m*p, p*n).
        # Flat index: (a * (m*p) + b) * (p*n) + c
        # Param index for U: a * r + r_idx
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    r_idx = a * (m*p) * (p*n) + b * (p*n) + c
                    for k in range(r):
                        # dR / dU[a, k] = V[b, k] * W[c, k]
                        J[r_idx, a * r + k] = V[b, k] * W[c, k]
                        
                        # dR / dV[b, k] = U[a, k] * W[c, k]
                        J[r_idx, sz_u + b * r + k] = U[a, k] * W[c, k]
                        
                        # dR / dW[c, k] = U[a, k] * V[b, k]
                        J[r_idx, sz_u + sz_v + c * r + k] = U[a, k] * V[b, k]
        return J

    def _is_exact(U, V, W, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_approx - T)) < tol

    def _try_rank(rank, n_restarts, rng, max_iter=1000):
        """Try to find exact decomposition at given rank."""
        if _remaining() < 10:
            return None
        
        for restart in range(n_restarts):
            if _remaining() < 5:
                return None
            
            # Multiple initialization strategies
            if restart % 3 == 0:
                # Random small
                U = rng.standard_normal((n*m, rank)) * 0.3
                V = rng.standard_normal((m*p, rank)) * 0.3
                W = rng.standard_normal((p*n, rank)) * 0.3
            elif restart % 3 == 1:
                # Random moderate
                U = rng.standard_normal((n*m, rank)) * 1.0
                V = rng.standard_normal((m*p, rank)) * 1.0
                W = rng.standard_normal((p*n, rank)) * 1.0
            else:
                # Perturb schoolbook (if rank <= rank_sb)
                if rank <= rank_sb:
                    U = U0[:, :rank].copy() + rng.standard_normal((n*m, rank)) * 0.1
                    V = V0[:, :rank].copy() + rng.standard_normal((m*p, rank)) * 0.1
                    W = W0[:, :rank].copy() + rng.standard_normal((p*n, rank)) * 0.1
                else:
                    U = rng.standard_normal((n*m, rank)) * 0.5
                    V = rng.standard_normal((m*p, rank)) * 0.5
                    W = rng.standard_normal((p*n, rank)) * 0.5
            
            params = _flatten_params(U, V, W, rank)
            
            try:
                res = minimize(
                    _objective, params, args=(rank,),
                    jac=lambda p, r: _objective_grad(p, r),
                    method='L-BFGS-B',
                    options={'maxiter': max_iter, 'gtol': 1e-12, 'ftol': 1e-14}
                )
                
                if _remaining() < 5:
                    return None
                
                if res.fun < 1e-10:
                    U_f, V_f, W_f = _unflatten(res.x, rank)
                    if _is_exact(U_f, V_f, W_f, tol=1e-9):
                        # Try to round to half-integers
                        U_r = np.round(U_f * 2) / 2
                        V_r = np.round(V_f * 2) / 2
                        W_r = np.round(W_f * 2) / 2
                        if _is_exact(U_r, V_r, W_r, tol=1e-9):
                            return U_r, V_r, W_r
                        else:
                            return U_f, V_f, W_f
            except Exception:
                pass
        
        return None
    
    # Try ranks from 17 down to 13
    rng = np.random.default_rng(42)
    
    for rank in range(rank_sb - 1, 12, -1):
        if _remaining() < 15:
            break
        
        # Try multiple restarts for each rank
        # More restarts for lower ranks (higher priority)
        if rank == rank_sb - 1:
            n_restarts = min(50, max(20, int(_remaining() / 2)))
        else:
            n_restarts = min(30, max(10, int(_remaining() / 3)))
        result = _try_rank(rank, n_restarts, rng)
        
        if result is not None:
            U_f, V_f, W_f = result
            best_U, best_V, best_W = U_f, V_f, W_f
            best_rank = rank
            # Found a valid decomposition; try to go lower
            continue
    
    # If we found a better rank, return it
    if best_rank < rank_sb:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        # Verify
        U_v = np.array(U_list)
        V_v = np.array(V_list)
        W_v = np.array(W_list)
        T_approx = np.einsum('ar,br,cr->abc', U_v, V_v, W_v)
        if np.max(np.abs(T_approx - T)) < 1e-9:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U_list = [[int(x) for x in row] for row in U0]
    V_list = [[int(x) for x in row] for row in V0]
    W_list = [[int(x) for x in row] for row in W0]
    return {"U": U_list, "V": V_list, "W": W_list}
