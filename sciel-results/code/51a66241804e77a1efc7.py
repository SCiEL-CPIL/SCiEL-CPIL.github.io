"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24
    
    # Schoolbook solution as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Search for lower ranks, starting from 13 (Strassen-like bound) up to 23
    # We prioritize lower ranks as they are the target for improvement
    for rank in range(13, 24):
        if time.time() > deadline:
            break
        
        try:
            result = _try_rank_strassen(n, m, p, rank, deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                break # Found a better rank, stop
        except Exception:
            continue
    
    return best_result

def _verify_solution(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                max_err = max(max_err, abs(val - 1.0))
    return max_err < tol

def _snap_to_half_integers(U, V, W, n, m, p):
    """Try snapping to half-integers and verify."""
    for denom in [1, 2, 4, 8]:
        U_snap = np.round(U * denom) / denom
        V_snap = np.round(V * denom) / denom
        W_snap = np.round(W * denom) / denom
        
        if _verify_solution(U_snap, V_snap, W_snap, n, m, p):
            return U_snap, V_snap, W_snap
    return None, None, None

def _try_rank_strassen(n, m, p, rank, deadline):
    """
    Try to find a rank decomposition with the given rank.
    Uses Levenberg-Marquardt ('lm') for speed and structured initialization.
    """
    um = n * m
    vm = m * p
    wm = p * n
    
    def residual_function(params):
        U = params[:um*rank].reshape(um, rank)
        V = params[um*rank:um*rank + vm*rank].reshape(vm, rank)
        W = params[um*rank + vm*rank:].reshape(wm, rank)
        
        residuals = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    residuals[idx] = val - 1.0
                    idx += 1
        return residuals

    np.random.seed(42 + rank)
    
    # More restarts for lower ranks to escape local minima
    num_restarts = 100 if rank < 15 else 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Structured initialization: 
        # Try to mimic Strassen's 7 terms for the 2x2x2 core, padded
        # For simplicity, we use random sparse initialization with 1s and -1s
        # and some zeros to encourage sparsity
        
        U0 = np.zeros((um, rank))
        V0 = np.zeros((vm, rank))
        W0 = np.zeros((wm, rank))
        
        # Initialize with random sparse entries
        for r in range(rank):
            # Randomly select a few entries to be non-zero
            u_idx = np.random.choice(um, size=min(um, 3), replace=False)
            v_idx = np.random.choice(vm, size=min(vm, 3), replace=False)
            w_idx = np.random.choice(wm, size=min(wm, 3), replace=False)
            
            U0[u_idx, r] = np.random.choice([-1, 1])
            V0[v_idx, r] = np.random.choice([-1, 1])
            W0[w_idx, r] = np.random.choice([-1, 1])
            
            # Add some 0.5 values
            if np.random.rand() > 0.5:
                U0[u_idx[0], r] = 0.5
                V0[v_idx[0], r] = 0.5
                W0[w_idx[0], r] = 0.5
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            # Use 'lm' (Levenberg-Marquardt) for faster convergence
            # If 'lm' fails due to shape, fallback to 'trf'
            result = least_squares(residual_function, params0, method='lm', max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        except Exception:
            try:
                result = least_squares(residual_function, params0, method='trf', max_nfev=200, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            except Exception:
                continue
            
        if result.cost < 1e-10:
            U_opt = result.x[:um*rank].reshape(um, rank)
            V_opt = result.x[um*rank:um*rank + vm*rank].reshape(vm, rank)
            W_opt = result.x[um*rank + vm*rank:].reshape(wm, rank)
            
            U_snap, V_snap, W_snap = _snap_to_half_integers(U_opt, V_opt, W_opt, n, m, p)
            if U_snap is not None:
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
    
    return None

# EVOLVE-BLOCK-END
