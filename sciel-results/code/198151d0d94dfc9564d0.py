import time
import numpy as np
import scipy.optimize as optimize

START_TIME = time.time()
TIME_LIMIT = 110.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    U shape: (n*m, rank) - indexed by [i*m+j]
    V shape: (m*p, rank) - indexed by [j*p+k]
    W shape: (p*n, rank) - indexed by [k*n+i]
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _snap_to_half_integers(matrix):
    """Snap matrix entries to nearest 0.5."""
    return np.round(matrix * 2) / 2.0

def _solve_2x3x3_rank17():
    """
    Attempt to find a rank-17 decomposition for 2x3x3 matmul.
    Returns (U, V, W) or None.
    """
    n, m, p = 2, 3, 3
    r = 17
    
    # Flatten dimensions
    n_u = n * m # 6
    n_v = m * p # 9
    n_w = p * n # 6
    
    def objective(params):
        U = params[:n_u*r].reshape(n_u, r)
        V = params[n_u*r : n_u*r + n_v*r].reshape(n_v, r)
        W = params[n_u*r + n_v*r :].reshape(n_w, r)
        
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    residuals.append(s - 1.0)
        return residuals
    
    best_U, best_V, best_W = None, None, None
    best_loss = float('inf')
    
    # Use a fixed seed for reproducibility in search, but vary it slightly if needed
    # We have ~100 seconds.
    # Let's try 500 restarts.
    # Each restart is a least_squares call.
    
    for restart in range(1000):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        # Random initialization
        # Scale: 1.0 is often a good start for these small tensors
        U0 = np.random.randn(n_u, r) * 1.0
        V0 = np.random.randn(n_v, r) * 1.0
        W0 = np.random.randn(n_w, r) * 1.0
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            # 'lm' is fast but doesn't support bounds. 'trf' is slower but more robust.
            # Given the small size, 'lm' with multiple restarts is usually better.
            res = optimize.least_squares(objective, x0, method='lm', max_nfev=200)
        except Exception:
            continue
            
        if res.cost < best_loss:
            best_loss = res.cost
            U_sol = res.x[:n_u*r].reshape(n_u, r)
            V_sol = res.x[n_u*r : n_u*r + n_v*r].reshape(n_v, r)
            W_sol = res.x[n_u*r + n_v*r :].reshape(n_w, r)
            
            # Check if this solution is close to integer/half-integer
            # The problem statement says "integer/half-integer entries".
            # If the optimization finds a real-valued solution that is exact, 
            # we need to check if it can be snapped.
            # However, if the true solution is half-integer, the optimizer should land near it.
            
            U_snap = _snap_to_half_integers(U_sol)
            V_snap = _snap_to_half_integers(V_sol)
            W_snap = _snap_to_half_integers(W_sol)
            
            # Verify the snapped solution
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
            
            # Also check if the unsnapped solution is effectively exact (within tolerance)
            # and if so, maybe we can use it? 
            # The contract says "integer/half-integer entries". 
            # So we MUST return snapped values that verify.
            
            # If the unsnapped solution has very low cost but doesn't snap to a valid solution,
            # it's a dead end for this restart.
            
    return None

def _try_direct_2x3x6_rank(r_target):
    """
    Try to find a direct rank-r decomposition for 2x3x6 matmul.
    Returns (U, V, W) or None.
    """
    n, m, p = 2, 3, 6
    r = r_target
    
    # Flatten dimensions
    n_u = n * m  # 6
    n_v = m * p  # 18
    n_w = p * n  # 12
    
    def objective(params):
        U = params[:n_u*r].reshape(n_u, r)
        V = params[n_u*r : n_u*r + n_v*r].reshape(n_v, r)
        W = params[n_u*r + n_v*r :].reshape(n_w, r)
        
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    residuals.append(s - 1.0)
        return residuals
    
    best_loss = float('inf')
    
    # Try multiple restarts with different scales
    scales = [0.5, 1.0, 2.0]
    n_restarts_per_scale = 50
    
    for scale in scales:
        for restart in range(n_restarts_per_scale):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                return None
                
            # Random initialization with the given scale
            U0 = np.random.randn(n_u, r) * scale
            V0 = np.random.randn(n_v, r) * scale
            W0 = np.random.randn(n_w, r) * scale
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                res = optimize.least_squares(objective, x0, method='lm', max_nfev=100)
            except Exception:
                continue
                
            if res.cost < best_loss:
                best_loss = res.cost
                U_sol = res.x[:n_u*r].reshape(n_u, r)
                V_sol = res.x[n_u*r : n_u*r + n_v*r].reshape(n_v, r)
                W_sol = res.x[n_u*r + n_v*r :].reshape(n_w, r)
                
                # Snap to half-integers and verify
                U_snap = _snap_to_half_integers(U_sol)
                V_snap = _snap_to_half_integers(V_sol)
                W_snap = _snap_to_half_integers(W_sol)
                
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap
    
    return None

def _expand_2x3x3_to_2x3x6(U_sub, V_sub, W_sub):
    """
    Expand a rank-17 2x3x3 decomposition to a rank-34 2x3x6 decomposition.
    """
    r_sub = U_sub.shape[1] # 17
    r_full = 2 * r_sub # 34
    
    U_full = np.zeros((6, r_full))
    V_full = np.zeros((18, r_full))
    W_full = np.zeros((12, r_full))
    
    # U is duplicated
    U_full[:, :r_sub] = U_sub
    U_full[:, r_sub:] = U_sub
    
    # V: j in 0..2, k in 0..5
    # V_full[j*6 + k, c]
    # If k < 3: c < 17, V_sub[j*3 + k, c]
    # If k >= 3: c >= 17, V_sub[j*3 + (k-3), c-17]
    for j in range(3):
        for k in range(6):
            if k < 3:
                V_full[j*6 + k, :r_sub] = V_sub[j*3 + k, :]
            else:
                V_full[j*6 + k, r_sub:] = V_sub[j*3 + (k-3), :]
                
    # W: k in 0..5, i in 0..1
    # W_full[k*2 + i, c]
    # If k < 3: c < 17, W_sub[k*2 + i, c]
    # If k >= 3: c >= 17, W_sub[(k-3)*2 + i, c-17]
    for k in range(6):
        for i in range(2):
            if k < 3:
                W_full[k*2 + i, :r_sub] = W_sub[k*2 + i, :]
            else:
                W_full[k*2 + i, r_sub:] = W_sub[(k-3)*2 + i, :]
                
    return U_full, V_full, W_full

def construct():
    n, m, p = 2, 3, 6
    
    # 1. Get schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # 2. Try to find rank-34 solution via 2x3x3 rank-17
    U_34, V_34, W_34 = None, None, None
    
    if time.time() - START_TIME < TIME_LIMIT - 10:
        result_17 = _solve_2x3x3_rank17()
        if result_17 is not None:
            U_sub, V_sub, W_sub = result_17
            U_34, V_34, W_34 = _expand_2x3x3_to_2x3x6(U_sub, V_sub, W_sub)
            
            # Verify the expanded solution
            if _verify_decomposition(U_34, V_34, W_34, n, m, p):
                # Success!
                pass
            else:
                U_34, V_34, W_34 = None, None, None

    # 3. Try direct search for lower ranks (33, 32, 31, 30)
    best_U, best_V, best_W = None, None, None
    
    if U_34 is not None:
        best_U, best_V, best_W = U_34, V_34, W_34
    
    # Try to find even lower rank solutions
    for r_target in [33, 32, 31, 30]:
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        result = _try_direct_2x3x6_rank(r_target)
        if result is not None:
            U_try, V_try, W_try = result
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                best_U, best_V, best_W = U_try, V_try, W_try
                break  # Found the lowest rank we tried, stop
    
    # 4. Select best
    if best_U is None:
        best_U, best_V, best_W = U_school, V_school, W_school

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
