import time
import numpy as np
from scipy.optimize import least_squares

_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 150s kill

def _time_remaining():
    return _DEADLINE - time.time()

def _get_target_tensor():
    n, m, p = 2, 2, 7
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _verify(U, V, W, T):
    T_rec = np.zeros_like(T)
    for r in range(U.shape[1]):
        T_rec += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    return np.allclose(T_rec, T, atol=1e-9)

def _is_half_integer(X, tol=1e-6):
    return np.allclose(X * 2, np.round(X * 2), atol=tol)

def _construct_rank_25():
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen(col_start, k_start):
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        U[0, col_start+0] += 1; U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1; V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1; W[2*(k_start+1) + 1, col_start+0] += 1
        # M2 = (A10 + A11) * B0,ks
        U[2, col_start+1] += 1; U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1; W[2*(k_start+1) + 1, col_start+1] -= 1
        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1; V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1; W[2*(k_start+1) + 1, col_start+2] += 1
        # M4 = A11 * (B1,ks - B0,ks)
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1; V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1; W[2*k_start + 1, col_start+3] += 1
        # M5 = (A00 + A01) * B1,ks+1
        U[0, col_start+4] += 1; U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1; W[2*(k_start+1) + 0, col_start+4] += 1
        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        U[2, col_start+5] += 1; U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1; V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1
        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        U[1, col_start+6] += 1; U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1; V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
    
    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)
    
    col_start = 21
    U[0, col_start+0] += 1; V[7*0 + 6, col_start+0] += 1; W[2*6 + 0, col_start+0] += 1
    U[1, col_start+1] += 1; V[7*1 + 6, col_start+1] += 1; W[2*6 + 0, col_start+1] += 1
    U[2, col_start+2] += 1; V[7*0 + 6, col_start+2] += 1; W[2*6 + 1, col_start+2] += 1
    U[3, col_start+3] += 1; V[7*1 + 6, col_start+3] += 1; W[2*6 + 1, col_start+3] += 1
    
    return U, V, W

def _construct_rank_28():
    n, m, p = 2, 2, 7
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return U, V, W

def _als_search_rank(T, rank, max_iters=500, tol=1e-10, max_restarts=10):
    """
    Alternating Least Squares search for a rank-`rank` decomposition.
    """
    n, m, p = 2, 2, 7
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    T_flat = T.flatten()
    
    best_cost = np.inf
    best_U, best_V, best_W = None, None, None

    for restart in range(max_restarts):
        if _time_remaining() < 5.0:
            break
            
        # Initialize
        U = np.random.randn(dim_u, rank)
        V = np.random.randn(dim_v, rank)
        W = np.random.randn(dim_w, rank)
        
        # Normalize to prevent scale issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        cost = np.inf
        for it in range(max_iters):
            if _time_remaining() < 5.0:
                break
                
            # Update U
            # T_ijk = sum_r U_ir V_jr W_kr
            # For fixed V, W, this is linear in U.
            # Reshape T to (dim_u, dim_v*dim_w)
            # T[i, j*dim_w + k] = sum_r U[i, r] * (V[j, r] * W[k, r])
            # Let A[r, j*dim_w + k] = V[j, r] * W[k, r]
            # Then T_flat[i] = U[i, :] @ A.T
            # U[i, :] = T_flat[i] @ A.T @ (A @ A.T)^-1 ?
            # Actually, solve U @ A.T = T_flat for U.
            # U = T_flat @ (A @ A.T)^-1 @ A ? No.
            # T_flat (dim_u x 1) ? No, T_flat is flattened.
            # Let's use matrix form.
            # T_2d[i, (j, k)] = sum_r U[i, r] V[j, r] W[k, r]
            # Let M[r, (j, k)] = V[j, r] W[k, r]
            # Then T_2d = U @ M.T
            # So U = T_2d @ M.T @ (M @ M.T)^-1 ?
            # U @ M.T = T_2d
            # U = T_2d @ (M.T)^+ = T_2d @ (M @ M.T)^-1 @ M
            # We can solve U * M.T = T_2d using lstsq.
            # X = U, A = M.T, B = T_2d
            # U, residuals, rank, s = np.linalg.lstsq(M.T, T_2d, rcond=None)
            
            M = np.einsum('jr,kr->rjk', V, W).reshape(rank, dim_v * dim_w)
            T_2d = T.reshape(dim_u, dim_v * dim_w)
            U, _, _, _ = np.linalg.lstsq(M.T, T_2d, rcond=None)
            
            # Update V
            # T_ijk = sum_r U_ir V_jr W_kr
            # T_2d[j, (i, k)] = sum_r V[j, r] U[i, r] W[k, r]
            # Let M[r, (i, k)] = U[i, r] W[k, r]
            # T_2d_v = V @ M.T
            M = np.einsum('ir,kr->rik', U, W).reshape(rank, dim_u * dim_w)
            T_2d_v = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
            V, _, _, _ = np.linalg.lstsq(M.T, T_2d_v, rcond=None)
            
            # Update W
            # T_ijk = sum_r U_ir V_jr W_kr
            # T_2d[k, (i, j)] = sum_r W[k, r] U[i, r] V[j, r]
            # Let M[r, (i, j)] = U[i, r] V[j, r]
            # T_2d_w = W @ M.T
            M = np.einsum('ir,jr->rij', U, V).reshape(rank, dim_u * dim_v)
            T_2d_w = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
            W, _, _, _ = np.linalg.lstsq(M.T, T_2d_w, rcond=None)
            
            # Calculate cost
            T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
            cost = np.linalg.norm(T_rec - T)
            
            if cost < best_cost:
                best_cost = cost
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if cost < tol:
                break
                
    if best_cost < 1e-6:
        # Try to round to half-integers
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2
        if _verify(U_r, V_r, W_r, T):
            return U_r, V_r, W_r
        
        # Check if unrounded is effectively half-integer
        if _verify(best_U, best_V, best_W, T) and _is_half_integer(best_U) and _is_half_integer(best_V) and _is_half_integer(best_W):
            return best_U, best_V, best_W
            
        # Sometimes ALS finds a solution that is close to half-integers but not exact
        # We can try a local polish around the rounded solution
        U_p = U_r.copy()
        V_p = V_r.copy()
        W_p = W_r.copy()
        
        # Use least_squares for a few steps to polish
        def residuals(x):
            U = x[:dim_u*rank].reshape(dim_u, rank)
            V = x[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
            W = x[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
            T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
            return T_rec.flatten() - T_flat
        
        x0 = np.concatenate([U_p.flatten(), V_p.flatten(), W_p.flatten()])
        try:
            result = least_squares(residuals, x0, method='trf', max_nfev=2000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if result.cost < 1e-8:
                x_opt = result.x
                U_opt = x_opt[:dim_u*rank].reshape(dim_u, rank)
                V_opt = x_opt[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
                W_opt = x_opt[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
                U_opt_r = np.round(U_opt * 2) / 2
                V_opt_r = np.round(V_opt * 2) / 2
                W_opt_r = np.round(W_opt * 2) / 2
                if _verify(U_opt_r, V_opt_r, W_opt_r, T):
                    return U_opt_r, V_opt_r, W_opt_r
        except Exception:
            pass

    return None

def _search_rank(T, rank, U_init, V_init, W_init, max_restarts=5, per_restart_time=15.0):
    """Search for a rank-`rank` decomposition using least_squares with multiple restarts."""
    n, m, p = 2, 2, 7
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    T_flat = T.flatten()
    
    def residuals(x):
        U = x[:dim_u*rank].reshape(dim_u, rank)
        V = x[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = x[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        # Efficient residual: sum over rank of outer products
        T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
        return T_rec.flatten() - T_flat
    
    best = None
    for restart in range(max_restarts):
        if _time_remaining() < 5.0:
            break
        # Initialization
        if U_init is not None and U_init.shape[1] >= rank:
            U0 = U_init[:, :rank].copy()
            V0 = V_init[:, :rank].copy()
            W0 = W_init[:, :rank].copy()
        else:
            U0 = np.random.randn(dim_u, rank) * 0.5
            V0 = np.random.randn(dim_v, rank) * 0.5
            W0 = np.random.randn(dim_w, rank) * 0.5
        # Perturb
        scale = 0.1 if restart == 0 else 0.5
        U0 += np.random.randn(*U0.shape) * scale
        V0 += np.random.randn(*V0.shape) * scale
        W0 += np.random.randn(*W0.shape) * scale
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(residuals, x0, method='trf', max_nfev=8000, xtol=1e-14, ftol=1e-14, gtol=1e-14)
            if result.cost < 1e-6:
                x_opt = result.x
                U_opt = x_opt[:dim_u*rank].reshape(dim_u, rank)
                V_opt = x_opt[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
                W_opt = x_opt[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
                
                # High-precision polish: run another least_squares from the current solution
                if _time_remaining() > 10.0:
                    try:
                        result2 = least_squares(residuals, x_opt, method='trf', max_nfev=5000, xtol=1e-15, ftol=1e-15, gtol=1e-15)
                        if result2.cost < result.cost:
                            x_opt = result2.x
                            U_opt = x_opt[:dim_u*rank].reshape(dim_u, rank)
                            V_opt = x_opt[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
                            W_opt = x_opt[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
                    except Exception:
                        pass
                
                # Round to half-integers
                U_r = np.round(U_opt * 2) / 2
                V_r = np.round(V_opt * 2) / 2
                W_r = np.round(W_opt * 2) / 2
                if _verify(U_r, V_r, W_r, T):
                    return U_r, V_r, W_r
                # Also check unrounded
                if _verify(U_opt, V_opt, W_opt, T) and _is_half_integer(U_opt) and _is_half_integer(V_opt) and _is_half_integer(W_opt):
                    return U_opt, V_opt, W_opt
        except Exception:
            pass
    return None

def construct():
    if _time_remaining() < 5.0:
        U, V, W = _construct_rank_28()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    T = _get_target_tensor()
    
    # Rank 25 (known valid)
    U25, V25, W25 = _construct_rank_25()
    if not _verify(U25, V25, W25, T):
        # Should not happen, but fallback
        U28, V28, W28 = _construct_rank_28()
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Try rank 24 with ALS (faster) and LS (slower)
    if _time_remaining() > 30.0:
        # Attempt 1: ALS for Rank 24
        res24_als = _als_search_rank(T, 24, max_iters=200, tol=1e-9, max_restarts=20)
        if res24_als is not None:
            U, V, W = res24_als
            if _verify(U, V, W, T):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Attempt 2: LS for Rank 24 (using Rank 25 as init)
        if _time_remaining() > 20.0:
            res24_ls = _search_rank(T, 24, U25, V25, W25, max_restarts=2, per_restart_time=10.0)
            if res24_ls is not None:
                U, V, W = res24_ls
                if _verify(U, V, W, T):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U25.tolist(), "V": V25.tolist(), "W": W25.tolist()}
