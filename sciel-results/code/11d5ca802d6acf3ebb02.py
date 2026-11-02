"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import fsolve, least_squares

DIMENSIONS = [3, 4, 7]

# Precompute index mappings for fast tensor evaluation
# T_ijk = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
# We want T_ijk = 1 for all i in [0,n), j in [0,m), k in [0,p)

def _build_index_maps(n, m, p):
    """Build index arrays for fast vectorized tensor evaluation."""
    I = np.repeat(np.arange(n), m * p)
    J = np.tile(np.repeat(np.arange(m), p), n)
    K = np.tile(np.arange(p), n * m)
    # Row indices for U, V, W
    u_rows = I * m + J
    v_rows = J * p + K
    w_rows = K * n + I
    return u_rows, v_rows, w_rows

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    np.random.seed(42)
    
    # Precompute index maps for fast evaluation
    u_rows, v_rows, w_rows = _build_index_maps(n, m, p)
    N = n * m * p  # total number of tensor entries
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    def fast_residual(x, r):
        """Vectorized residual computation."""
        U = x[:r*n*m].reshape(n*m, r)
        V = x[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
        W = x[r*n*m + r*m*p:].reshape(p*n, r)
        
        # For each entry (i,j,k), compute sum_t U[u_row, t] * V[v_row, t] * W[w_row, t]
        U_sel = U[u_rows, :]  # (N, r)
        V_sel = V[v_rows, :]  # (N, r)
        W_sel = W[w_rows, :]  # (N, r)
        
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return vals - 1.0

    def fast_jacobian(x, r):
        """Analytical Jacobian for fast convergence."""
        nU = n * m * r
        nV = m * p * r
        nW = p * n * r
        n_vars = nU + nV + nW
        
        U = x[:nU].reshape(n*m, r)
        V = x[nU:nU+nV].reshape(m*p, r)
        W = x[nU+nV:].reshape(p*n, r)
        
        U_sel = U[u_rows, :]
        V_sel = V[v_rows, :]
        W_sel = W[w_rows, :]
        
        J = np.zeros((N, n_vars))
        
        # Precompute column indices for each tensor component
        # u_cols[i, t] is the column index in J corresponding to U[u_rows[i], t]
        u_cols = u_rows[:, None] * r + np.arange(r)[None, :]
        v_cols = v_rows[:, None] * r + np.arange(r)[None, :] + nU
        w_cols = w_rows[:, None] * r + np.arange(r)[None, :] + nU + nV
        
        # Compute partial derivatives
        # dT/dU = V * W
        # dT/dV = U * W
        # dT/dW = U * V
        dU = V_sel * W_sel
        dV = U_sel * W_sel
        dW = U_sel * V_sel
        
        # Fill Jacobian
        # Using np.put or advanced indexing
        # We loop over t because r is small (typically < 100)
        for t in range(r):
            J[np.arange(N), u_cols[:, t]] = dU[:, t]
            J[np.arange(N), v_cols[:, t]] = dV[:, t]
            J[np.arange(N), w_cols[:, t]] = dW[:, t]
            
        return J

    def verify(U, V, W):
        """Check if decomposition is exact."""
        U_sel = U[u_rows, :]
        V_sel = V[v_rows, :]
        W_sel = W[w_rows, :]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.max(np.abs(vals - 1.0))
    
    def compress_factors(U, V, W, r):
        """
        Greedy basis-reduction: iteratively identify and eliminate dependent rank-1 components.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        cur_r = r
        
        for _ in range(cur_r):
            if time.time() > deadline - 10:
                break
            
            best_remove = -1
            best_cost = np.inf
            
            for t in range(cur_r):
                if time.time() > deadline - 10:
                    break
                
                mask = np.ones(cur_r, dtype=bool)
                mask[t] = False
                
                U_red = U[:, mask]
                V_red = V[:, mask]
                W_red = W[:, mask]
                
                U_sel = U_red[u_rows, :]
                V_sel = V_red[v_rows, :]
                W_sel = W_red[w_rows, :]
                vals = np.sum(U_sel * V_sel * W_sel, axis=1)
                cost = np.sum((vals - 1.0) ** 2)
                
                if cost < best_cost:
                    best_cost = cost
                    best_remove = t
            
            if best_remove >= 0 and best_cost < 1e-10:
                mask = np.ones(cur_r, dtype=bool)
                mask[best_remove] = False
                U = U[:, mask]
                V = V[:, mask]
                W = W[:, mask]
                cur_r -= 1
            else:
                break
        
        return U, V, W, cur_r

    # Search from low rank upwards
    # Start from 40 to save time, as lower ranks are very difficult to find randomly
    min_rank_to_try = 40
    
    for r in range(min_rank_to_try, rank_school + 1):
        if time.time() > deadline - 20:
            break
        
        n_vars = r * (n*m + m*p + p*n)
        
        found = False
        n_inits = 5
        
        for init in range(n_inits):
            if time.time() > deadline - 15:
                break
            
            # Random initialization
            x0 = np.random.randn(n_vars) * 0.5
            
            try:
                result = least_squares(
                    fast_residual,
                    x0,
                    jac=lambda x: fast_jacobian(x, r),
                    args=(r,),
                    method='trf',
                    max_nfev=1000,
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12
                )
                
                if result.cost < 1e-10:
                    sol = result.x
                    U_sol = sol[:r*n*m].reshape(n*m, r)
                    V_sol = sol[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
                    W_sol = sol[r*n*m + r*m*p:].reshape(p*n, r)
                    
                    # Verify
                    max_err = verify(U_sol, V_sol, W_sol)
                    
                    if max_err < 1e-8:
                        # Try compression
                        U_comp, V_comp, W_comp, comp_r = compress_factors(U_sol, V_sol, W_sol, r)
                        
                        if comp_r < r:
                            # Verify compressed
                            max_err_comp = verify(U_comp, V_comp, W_comp)
                            if max_err_comp < 1e-8:
                                best_rank = comp_r
                                best_U, best_V, best_W = U_comp, V_comp, W_comp
                            else:
                                best_rank = r
                                best_U, best_V, best_W = U_sol, V_sol, W_sol
                        else:
                            best_rank = r
                            best_U, best_V, best_W = U_sol, V_sol, W_sol
                        
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    # Convert to half-integers if possible
    def to_half_integer(arr):
        return np.round(arr * 2) / 2
    
    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)
    
    # Verify rounded
    max_err = verify(U_out, V_out, W_out)
    
    if max_err > 1e-6:
        U_out, V_out, W_out = best_U, best_V, best_W
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
