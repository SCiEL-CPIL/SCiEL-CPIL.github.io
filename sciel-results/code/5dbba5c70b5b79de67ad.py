"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _build_true_tensor(n, m, p):
    """Build the true multiplication tensor T[a,b,c]."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify(U, V, W, T_true):
    """Verify that U, V, W exactly decompose the tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    return np.allclose(T_recon, T_true, atol=1e-10)

def _snap_to_half_int(arr, tol=1e-6):
    """Snap values to nearest half-integer if close enough."""
    snapped = np.round(arr * 2) / 2
    if np.allclose(arr, snapped, atol=tol):
        return snapped
    return arr

def _als_optimize(U, V, W, T_true, max_iters=2000, tol=1e-10, time_deadline=None):
    """ALS optimization for tensor decomposition."""
    n_m, r = U.shape
    m_p, _ = V.shape
    p_n, _ = W.shape
    
    for it in range(max_iters):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        # Update U: for each a, solve U[a,:] from linear system
        # T[a,:,:] = sum_c U[a,c] * (V[:,c] * W[:,c])
        # So U[a,:] @ (V*W_outer)^T = T[a,:,:].reshape(-1)
        # where (V*W_outer)[b,c] = V[b,c]*W[c,c]... no wait
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed a: T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])
        # Let M[b,c] = V[b,r]*W[c,r] for each r, then T[a,:,:] = sum_r U[a,r] * M_r
        # This is: vec(T[a,:,:]) = sum_r U[a,r] * vec(M_r)
        # So we can set up a linear system
        
        # Build the design matrix: for each r, vec(V[:,r] * W[:,r])
        # Shape: (m*p*p*n, r)
        # This is too large for direct construction, use einsum
        
        # Instead, use gradient-based approach with adaptive step size
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T_true - T_recon
        
        # Compute gradients
        grad_U = np.einsum('abc,br,cr->ar', R, V, W)
        grad_V = np.einsum('abc,ar,cr->br', R, U, W)
        grad_W = np.einsum('abc,ar,br->cr', R, U, V)
        
        # Adaptive learning rate
        err = np.linalg.norm(R)
        if err < tol:
            break
        
        # Use a simple adaptive step
        scale_U = np.linalg.norm(grad_U) / (np.linalg.norm(U) + 1e-12)
        scale_V = np.linalg.norm(grad_V) / (np.linalg.norm(V) + 1e-12)
        scale_W = np.linalg.norm(grad_W) / (np.linalg.norm(W) + 1e-12)
        
        eta_U = 0.1 / (scale_U + 1)
        eta_V = 0.1 / (scale_V + 1)
        eta_W = 0.1 / (scale_W + 1)
        
        U += eta_U * grad_U
        V += eta_V * grad_V
        W += eta_W * grad_W
    
    return U, V, W

def _try_rank_als(r, n, m, p, T_true, num_restarts=10, max_iters=1500, time_deadline=None):
    """Try to find a rank-r decomposition using ALS with multiple restarts."""
    best_err = float('inf')
    best_factors = None
    
    for restart in range(num_restarts):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        rng = np.random.RandomState(42 + restart)
        
        # Initialize with schoolbook-like structure plus noise
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Initialize: for each of the n*m*p "natural" terms, assign to a random column
        # But we only have r columns, so we need to be smart
        # Use random initialization
        U = rng.randn(n*m, r) * 0.5
        V = rng.randn(m*p, r) * 0.5
        W = rng.randn(p*n, r) * 0.5
        
        # Optimize
        U, V, W = _als_optimize(U, V, W, T_true, max_iters=max_iters, time_deadline=time_deadline)
        
        # Check error
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T_true))
        
        if err < best_err:
            best_err = err
            best_factors = (U.copy(), V.copy(), W.copy())
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-8:
        U, V, W = best_factors
        # Try to snap to half-integers
        U_snap = _snap_to_half_int(U)
        V_snap = _snap_to_half_int(V)
        W_snap = _snap_to_half_int(W)
        
        if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), T_true):
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
        
        # Try snapping with slightly looser tolerance
        U_snap = _snap_to_half_int(U, tol=1e-5)
        V_snap = _snap_to_half_int(V, tol=1e-5)
        W_snap = _snap_to_half_int(W, tol=1e-5)
        
        if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), T_true):
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
        
        # If close but not exact, do a final polish
        U_pol, V_pol, W_pol = _als_optimize(U_snap, V_snap, W_snap, T_true, max_iters=500, time_deadline=time_deadline)
        U_snap2 = _snap_to_half_int(U_pol, tol=1e-5)
        V_snap2 = _snap_to_half_int(V_pol, tol=1e-5)
        W_snap2 = _snap_to_half_int(W_pol, tol=1e-5)
        
        if _verify(U_snap2.tolist(), V_snap2.tolist(), W_snap2.tolist(), T_true):
            return U_snap2.tolist(), V_snap2.tolist(), W_snap2.tolist()
        
        # Return the best we have (might not be exact, but let's check)
        if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), T_true):
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
    
    return None

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x8 matrix.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    T_true = _build_true_tensor(n, m, p)
    
    # Schoolbook fallback
    schoolbook_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = schoolbook_result
    
    # Deadline for search
    search_deadline = START_TIME + TIME_LIMIT - 5
    
    # Try structured ranks
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But we can likely do better than schoolbook (96)
    
    # Try ranks in a strategic order
    candidate_ranks = []
    
    # First, try some "natural" structured ranks
    # For 3x4x8, some known constructions might give rank around 40-60
    
    # Try from a reasonable lower bound upward
    lb = max(n*m, m*p, p*n)  # 32
    
    # Add some strategic candidates
    for r in range(lb, best_rank):
        candidate_ranks.append(r)
    
    # Limit the number of trials to fit within time budget
    # Each trial takes roughly 0.5-2 seconds depending on rank
    max_trials = 30
    
    # Sort by likelihood of success (try mid-range first, then expand)
    # Actually, let's try a few specific ranges
    trial_ranks = []
    
    # Try ranks that are multiples of small numbers (structural)
    for r in range(lb, min(lb + 40, best_rank)):
        trial_ranks.append(r)
    
    # Also try some higher ranks if we have time
    if len(trial_ranks) < max_trials:
        for r in range(lb + 40, min(lb + 60, best_rank)):
            if len(trial_ranks) < max_trials:
                trial_ranks.append(r)
    
    trial_ranks = trial_ranks[:max_trials]
    
    for r in trial_ranks:
        if time.time() > search_deadline:
            break
        if r >= best_rank:
            break
        
        result = _try_rank_als(r, n, m, p, T_true, num_restarts=5, max_iters=1000, time_deadline=search_deadline)
        
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, T_true):
                best_rank = r
                best_result = {"U": U, "V": V, "W": W}
                # Found a valid decomposition, try to go lower
                # Remove all ranks >= r from trial list
                trial_ranks = [x for x in trial_ranks if x < r]
                # Add some lower ranks to try
                for new_r in range(lb, r):
                    if new_r not in trial_ranks:
                        trial_ranks.append(new_r)
                trial_ranks.sort()
                break  # Restart the loop with updated trial_ranks
    
    # Final check: if we found something better, use it; otherwise schoolbook
    if best_rank < n * m * p:
        return best_result
    else:
        return schoolbook_result
# EVOLVE-BLOCK-END
