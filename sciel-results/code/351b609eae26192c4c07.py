"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Starts with Strassen's rank-42 solution and attempts to reduce rank via
    numerical optimization and rank-reduction heuristics.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    t_start = time.time()
    deadline = t_start + 115  # safety margin
    
    # First, build the Strassen rank-42 solution as baseline
    num_pairs = p // 2  # 6 pairs
    rank_base = num_pairs * 7  # 42
    
    U_base = np.zeros((n * m, rank_base))
    V_base = np.zeros((m * p, rank_base))
    W_base = np.zeros((p * n, rank_base))
    
    for t in range(num_pairs):
        base = t * 7
        for s in range(7):
            c = base + s
            if s == 0:
                U_base[0, c] = 1; U_base[3, c] = 1
                V_base[2*t, c] = 1; V_base[p + 2*t+1, c] = 1
                W_base[2*t*n + 0, c] = 1; W_base[(2*t+1)*n + 1, c] = 1
            elif s == 1:
                U_base[2, c] = 1; U_base[3, c] = 1
                V_base[2*t, c] = 1
                W_base[2*t*n + 1, c] = 1; W_base[(2*t+1)*n + 1, c] = -1
            elif s == 2:
                U_base[0, c] = 1
                V_base[2*t+1, c] = 1; V_base[p + 2*t+1, c] = -1
                W_base[(2*t+1)*n + 0, c] = 1; W_base[(2*t+1)*n + 1, c] = 1
            elif s == 3:
                U_base[3, c] = 1
                V_base[p + 2*t, c] = 1; V_base[2*t, c] = -1
                W_base[2*t*n + 0, c] = 1; W_base[2*t*n + 1, c] = 1
            elif s == 4:
                U_base[0, c] = 1; U_base[1, c] = 1
                V_base[p + 2*t+1, c] = 1
                W_base[2*t*n + 0, c] = -1; W_base[(2*t+1)*n + 0, c] = 1
            elif s == 5:
                U_base[2, c] = 1; U_base[0, c] = -1
                V_base[2*t, c] = 1; V_base[2*t+1, c] = 1
                W_base[(2*t+1)*n + 1, c] = 1
            elif s == 6:
                U_base[1, c] = 1; U_base[3, c] = -1
                V_base[p + 2*t, c] = 1; V_base[p + 2*t+1, c] = 1
                W_base[2*t*n + 0, c] = 1
    
    def verify(U, V, W, tol=1e-8):
        """Check if decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Verify baseline
    if not verify(U_base, V_base, W_base):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school))
        V_s = np.zeros((m * p, rank_school))
        W_s = np.zeros((p * n, rank_school))
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    c = ii*m + jj*p + kk
                    U_s[ii*m+jj, c] = 1
                    V_s[jj*p+kk, c] = 1
                    W_s[kk*n+ii, c] = 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    best_rank = rank_base
    best_U, best_V, best_W = U_base, V_base, W_base
    
    # Try to reduce rank by zeroing out columns and re-optimizing
    # Strategy: try to remove one column at a time and see if we can
    # re-optimize the remaining columns to still satisfy the constraints
    
    def try_reduce_rank(U, V, W, target_rank, max_iter=50):
        """Try to find a valid decomposition with target_rank by optimizing."""
        if target_rank >= U.shape[1]:
            return None
        
        # Start from current solution, zero out columns to get to target_rank
        U_r = U[:, :target_rank].copy()
        V_r = V[:, :target_rank].copy()
        W_r = W[:, :target_rank].copy()
        
        # Flatten parameters
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten(x):
            U = x[:U_r.size].reshape(U_r.shape)
            V = x[U_r.size:U_r.size+V_r.size].reshape(V_r.shape)
            W = x[U_r.size+V_r.size:].reshape(W_r.shape)
            return U, V, W
        
        def objective(x):
            U, V, W = unflatten(x)
            # Compute all 48 constraints
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        errors.append(val - 1.0)
            return np.sum(np.array(errors)**2)
        
        x0 = flatten(U_r, V_r, W_r)
        
        # Use Nelder-Mead for small problems
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10})
            if result.fun < 1e-8:
                U_opt, V_opt, W_opt = unflatten(result.x)
                if verify(U_opt, V_opt, W_opt, tol=1e-7):
                    return U_opt, V_opt, W_opt
        except Exception:
            pass
        
        return None
    
    # Try reducing rank one at a time
    current_U, current_V, current_W = U_base, V_base, W_base
    current_rank = rank_base
    
    for target_rank in range(current_rank - 1, max(current_rank - 5, 15), -1):
        if time.time() > deadline - 20:
            break
        
        result = try_reduce_rank(current_U, current_V, current_W, target_rank, max_iter=100)
        if result is not None:
            current_U, current_V, current_W = result
            current_rank = target_rank
            best_rank = target_rank
            best_U, best_V, best_W = current_U, current_V, current_W
    
    # Convert to lists with integer/half-integer rounding if close
    def round_if_close(arr):
        """Round values that are close to half-integers."""
        rounded = np.round(arr * 2) / 2
        if np.allclose(arr, rounded, atol=1e-6):
            return rounded
        return arr
    
    best_U = round_if_close(best_U)
    best_V = round_if_close(best_V)
    best_W = round_if_close(best_W)
    
    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-6):
        # Use the original Strassen solution
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
