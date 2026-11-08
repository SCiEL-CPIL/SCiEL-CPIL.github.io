import time
import numpy as np
from scipy.optimize import least_squares

_START_TIME = time.time()
_TIME_LIMIT = 130.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _elapsed()

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    r = U.shape[1]
    if r == 0:
        return False
    T = np.ones((n, m, p), dtype=np.float64)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T_computed = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T_computed, T, atol=tol)

def _strassen_28():
    """
    Build Strassen-based decomposition for C = A * B where A is 2x4, B is 4x4, C is 2x4.
    Returns rank 28.
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    U_coeffs = [
        [(0,0,1), (1,1,1)],      # M0
        [(1,0,1), (1,1,1)],      # M1
        [(0,0,1)],               # M2
        [(1,1,1)],               # M3
        [(0,0,1), (0,1,1)],      # M4
        [(1,0,1), (0,0,-1)],     # M5
        [(0,1,1), (1,1,-1)],     # M6
    ]
    
    V_coeffs = [
        [(0,0,1), (1,1,1)],      # M0
        [(0,0,1)],               # M1
        [(0,1,1), (1,1,-1)],     # M2
        [(1,0,1), (0,0,-1)],     # M3
        [(1,1,1)],               # M4
        [(0,0,1), (0,1,1)],      # M5
        [(1,0,1), (1,1,1)],      # M6
    ]
    
    W_coeff_map = {
        0: [(0,0,1), (1,1,1)],    # M0
        1: [(1,0,1), (1,1,-1)],   # M1
        2: [(0,1,1), (1,1,1)],    # M2
        3: [(0,0,1), (1,0,1)],    # M3
        4: [(0,0,-1), (0,1,1)],   # M4
        5: [(1,1,1)],             # M5
        6: [(0,0,1)],             # M6
    }
    
    col = 0
    for jp in range(2):
        for kp in range(2):
            for t in range(7):
                for (i_l, j_l, coeff) in U_coeffs[t]:
                    i = i_l
                    j = jp * 2 + j_l
                    U[i * m + j, col] += coeff
                for (j_l, k_l, coeff) in V_coeffs[t]:
                    j = jp * 2 + j_l
                    k = kp * 2 + k_l
                    V[j * p + k, col] += coeff
                for (i_l, k_l, coeff) in W_coeff_map[t]:
                    i = i_l
                    k = kp * 2 + k_l
                    W[k * n + i, col] += coeff
                col += 1
    
    return U, V, W

def _compute_tensor(U, V, W, n, m, p):
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    return np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)

def _optimize_decomposition(U_init, V_init, W_init, n, m, p, fixed_mask=None, time_budget=5.0):
    """
    Optimizes the non-fixed entries of U, V, W to minimize ||T_computed - T_target||^2.
    fixed_mask: boolean array of shape (rank,) indicating which columns are fixed (True) or free (False).
    If fixed_mask is None, all columns are free.
    """
    r = U_init.shape[1]
    T_target = np.ones((n, m, p), dtype=np.float64)
    
    if fixed_mask is None:
        fixed_mask = np.zeros(r, dtype=bool)
        
    # Indices of free columns
    free_cols = np.where(~fixed_mask)[0]
    
    if len(free_cols) == 0:
        return U_init, V_init, W_init, 0.0

    # Flatten the free parts of U, V, W
    # U shape: (n*m, r). Free columns: U[:, free_cols]
    # We need to map flat index to (row, col)
    
    u_free = U_init[:, free_cols].copy().flatten()
    v_free = V_init[:, free_cols].copy().flatten()
    w_free = W_init[:, free_cols].copy().flatten()
    
    # Combine into one vector
    x0 = np.concatenate([u_free, v_free, w_free])
    
    size_u = len(u_free)
    size_v = len(v_free)
    
    def residual(x):
        u_flat = x[:size_u]
        v_flat = x[size_u:size_u+size_v]
        w_flat = x[size_u+size_v:]
        
        # Reconstruct U, V, W with fixed and free parts
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        U[:, free_cols] = u_flat.reshape(n*m, -1)
        V[:, free_cols] = v_flat.reshape(m*p, -1)
        W[:, free_cols] = w_flat.reshape(p*n, -1)
        
        T_comp = _compute_tensor(U, V, W, n, m, p)
        return (T_comp - T_target).flatten()

    start_time = time.time()
    try:
        # Use least_squares for robustness
        # Method 'trf' (trust region reflective) is good for bound constraints, but we don't have bounds.
        # 'lm' (Levenberg-Marquardt) is fast for small residuals but can fail if far from solution.
        # Let's use 'trf' with max_nfev to limit time.
        # max_nfev = 100 is usually fast.
        result = least_squares(residual, x0, method='trf', max_nfev=100)
        
        # Extract best x
        x_best = result.x
        u_best = x_best[:size_u].reshape(n*m, -1)
        v_best = x_best[size_u:size_u+size_v].reshape(m*p, -1)
        w_best = x_best[size_u+size_v:].reshape(p*n, -1)
        
        U_best = U_init.copy()
        V_best = V_init.copy()
        W_best = W_init.copy()
        
        U_best[:, free_cols] = u_best
        V_best[:, free_cols] = v_best
        W_best[:, free_cols] = w_best
        
        err = np.linalg.norm(result.fun)
        
        return U_best, V_best, W_best, err
        
    except Exception:
        return U_init, V_init, W_init, np.inf

def _reduce_rank_optimized(U, V, W, n, m, p):
    """
    Attempts to reduce rank by removing one column at a time and re-optimizing.
    """
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    r = current_U.shape[1]
    
    # We will try to reduce rank from r down to 15
    target_rank = 15
    
    while r > target_rank and _remaining() > 10.0:
        improved = False
        # Try removing each column
        for c in range(r):
            if _remaining() < 5.0:
                break
            
            # Create mask: all True (fixed) except c is False (free to become 0? No, we want to remove it)
            # Strategy: Fix all columns EXCEPT c to their current values?
            # No, if we remove column c, we set it to 0.
            # We want to see if the remaining r-1 columns can compensate.
            # So we fix the other r-1 columns? No, we should optimize them too to allow compensation.
            # But if we optimize them, we might drift far from the known good solution.
            # Better: Fix the other r-1 columns to their current values, and only optimize... wait.
            # If we fix the other columns, we can't change them to compensate.
            # So we must allow the other columns to change.
            # But we must force column c to be 0.
            
            # So:
            # 1. Set column c of U, V, W to 0.
            # 2. Mark column c as FIXED (so it stays 0).
            # 3. Mark all other columns as FREE (so they can adjust).
            # 4. Optimize.
            
            U_test = current_U.copy()
            V_test = current_V.copy()
            W_test = current_W.copy()
            
            # Zero out column c
            U_test[:, c] = 0
            V_test[:, c] = 0
            W_test[:, c] = 0
            
            # Mask: True means fixed.
            # We want column c to be fixed at 0.
            # We want other columns to be free.
            fixed_mask = np.ones(r, dtype=bool)
            fixed_mask[c] = True  # Fixed
            # Others are False (Free)
            
            # Wait, if we fix column c to 0, and free others, the optimizer will try to make the others compensate.
            # This is correct.
            
            U_opt, V_opt, W_opt, err = _optimize_decomposition(
                U_test, V_test, W_test, n, m, p, 
                fixed_mask=fixed_mask, 
                time_budget=2.0
            )
            
            if err < 1e-8:
                # Success! We found a valid decomposition with rank r-1.
                # Remove column c
                mask = [i for i in range(r) if i != c]
                current_U = U_opt[:, mask]
                current_V = V_opt[:, mask]
                current_W = W_opt[:, mask]
                r = r - 1
                improved = True
                break
        
        if not improved:
            # Could not reduce rank by 1
            break
            
    return current_U, current_V, current_W

def _to_list_of_lists(mat):
    return [[float(x) for x in row] for row in mat]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = 2, 4, 4
    
    if _remaining() < 10.0:
        U, V, W = _schoolbook(n, m, p)
        return {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}
    
    # Build Strassen decomposition
    U, V, W = _strassen_28()
    
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}
    
    # Try to reduce rank using optimization
    U_red, V_red, W_red = _reduce_rank_optimized(U, V, W, n, m, p)
    
    # Final verification
    if _verify(U_red, V_red, W_red, n, m, p):
        return {"U": _to_list_of_lists(U_red), "V": _to_list_of_lists(V_red), "W": _to_list_of_lists(W_red)}
    else:
        # Fallback to original 28-term
        return {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]) if result["U"] else 0)
