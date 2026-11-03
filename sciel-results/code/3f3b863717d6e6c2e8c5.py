import time
import numpy as np
import scipy.optimize as optimize

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition of matmul tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None

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
    """Snap values to nearest half-integer."""
    return np.round(matrix * 2) / 2.0

def _snap_to_integers(matrix):
    """Snap values to nearest integer."""
    return np.round(matrix)

def _compute_matricization_ranks(n, m, p):
    """Compute the rank of each mode matricization of the matmul tensor.
    
    The matmul tensor T[i,j,k] = 1 for all valid (i,j,k).
    Mode-1 matricization: reshape to (n, m*p)
    Mode-2 matricization: reshape to (m, n*p)  
    Mode-3 matricization: reshape to (p, n*m)
    
    The tensor rank >= max(rank of each matricization).
    """
    # Build the full tensor
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Mode-1 matricization: (n, m*p)
    T1 = T.reshape(n, m * p)
    rank1 = np.linalg.matrix_rank(T1, tol=1e-10)
    
    # Mode-2 matricization: (m, n*p)
    T2 = T.transpose(1, 0, 2).reshape(m, n * p)
    rank2 = np.linalg.matrix_rank(T2, tol=1e-10)
    
    # Mode-3 matricization: (p, n*m)
    T3 = T.transpose(2, 0, 1).reshape(p, n * m)
    rank3 = np.linalg.matrix_rank(T3, tol=1e-10)
    
    lower_bound = max(rank1, rank2, rank3)
    return lower_bound, (rank1, rank2, rank3)

def _build_residual_vector(U, V, W, n, m, p, r):
    """Build the residual vector for the decomposition constraints."""
    residuals = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                residuals[idx] = s - 1.0
                idx += 1
    return residuals

def _try_rank_r(n, m, p, r, max_restarts=50, time_budget=15.0):
    """Try to find a rank-r decomposition using least squares optimization.
    
    Uses multiple random restarts and snaps to half-integers for exactness.
    """
    n_eq = n * m * p
    n_vars = (n*m + m*p + p*n) * r
    
    def objective(params):
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r :].reshape(p*n, r)
        return _build_residual_vector(U, V, W, n, m, p, r)
    
    best_U, best_V, best_W = None, None, None
    best_loss = float('inf')
    
    # Try different initialization strategies
    init_strategies = [
        'random_small', 'random_large', 'structured', 'schoolbook_subsample'
    ]
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > time_budget:
            break
        
        strategy = init_strategies[restart % len(init_strategies)]
        
        if strategy == 'random_small':
            U0 = np.random.randn(n*m, r) * 0.3
            V0 = np.random.randn(m*p, r) * 0.3
            W0 = np.random.randn(p*n, r) * 0.3
        elif strategy == 'random_large':
            U0 = np.random.randn(n*m, r) * 1.0
            V0 = np.random.randn(m*p, r) * 1.0
            W0 = np.random.randn(p*n, r) * 1.0
        elif strategy == 'structured':
            # Initialize with a structured pattern based on tensor symmetries
            U0 = np.zeros((n*m, r))
            V0 = np.zeros((m*p, r))
            W0 = np.zeros((p*n, r))
            for c in range(r):
                # Spread nonzeros across the factors
                for idx_u in range(n*m):
                    if (idx_u + c) % 3 == 0:
                        U0[idx_u, c] = 1.0 / np.sqrt(3)
                for idx_v in range(m*p):
                    if (idx_v + c) % 3 == 0:
                        V0[idx_v, c] = 1.0 / np.sqrt(3)
                for idx_w in range(p*n):
                    if (idx_w + c) % 3 == 0:
                        W0[idx_w, c] = 1.0 / np.sqrt(3)
        else:  # schoolbook_subsample
            # Start from schoolbook and randomly zero out some entries
            U0, V0, W0 = _schoolbook(n, m, p)
            U0 = U0[:, :min(r, n*m*p)]
            V0 = V0[:, :min(r, m*p)]
            W0 = W0[:, :min(r, p*n)]
            if U0.shape[1] < r:
                U0 = np.hstack([U0, np.random.randn(n*m, r - U0.shape[1]) * 0.1])
            if V0.shape[1] < r:
                V0 = np.hstack([V0, np.random.randn(m*p, r - V0.shape[1]) * 0.1])
            if W0.shape[1] < r:
                W0 = np.hstack([W0, np.random.randn(p*n, r - W0.shape[1]) * 0.1])
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = optimize.least_squares(objective, x0, method='lm', max_nfev=300)
        except Exception:
            continue
            
        if res.cost < best_loss:
            best_loss = res.cost
            U_sol = res.x[:n*m*r].reshape(n*m, r)
            V_sol = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_sol = res.x[n*m*r + m*p*r :].reshape(p*n, r)
            
            # Try snapping to half-integers
            U_snap = _snap_to_half_integers(U_sol)
            V_snap = _snap_to_half_integers(V_sol)
            W_snap = _snap_to_half_integers(W_sol)
            
            ok, _ = _verify_decomposition(U_snap, V_snap, W_snap, n, m, p)
            if ok:
                return U_snap, V_snap, W_snap
            
            # Try snapping to integers
            U_snap_int = _snap_to_integers(U_sol)
            V_snap_int = _snap_to_integers(V_sol)
            W_snap_int = _snap_to_integers(W_sol)
            
            ok_int, _ = _verify_decomposition(U_snap_int, V_snap_int, W_snap_int, n, m, p)
            if ok_int:
                return U_snap_int, V_snap_int, W_snap_int
            
            # Try the un-snapped solution with tighter verification
            ok_raw, _ = _verify_decomposition(U_sol, V_sol, W_sol, n, m, p, tol=1e-8)
            if ok_raw:
                # The solution is exact in floating point, but we need half-integers
                # Try a local refinement around this solution
                best_U, best_V, best_W = U_sol.copy(), V_sol.copy(), W_sol.copy()
    
    return None

def _try_direct_construction(n, m, p, r):
    """Try to directly construct a rank-r decomposition using algebraic methods.
    
    For small dimensions, we can try to solve the system directly.
    """
    # For the matmul tensor, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is a system of n*m*p equations in (n*m + m*p + p*n)*r unknowns
    # For small r, we can try to solve it using a different approach
    
    # Try a rank-1 peeling approach: greedily find rank-1 components
    T = np.ones((n, m, p), dtype=np.float64)
    U_list = []
    V_list = []
    W_list = []
    
    remaining = T.copy()
    for c in range(r):
        if time.time() - START_TIME > 145:
            break
        
        # Find the best rank-1 approximation of remaining
        # This is a non-convex problem, so we use a heuristic
        best_loss = float('inf')
        best_u, best_v, best_w = None, None, None
        
        # Try several random initializations for rank-1
        for trial in range(20):
            u = np.random.randn(n*m)
            v = np.random.randn(m*p)
            w = np.random.randn(p*n)
            
            # Normalize to avoid scale issues
            u_norm = np.linalg.norm(u)
            v_norm = np.linalg.norm(v)
            w_norm = np.linalg.norm(w)
            if u_norm < 1e-10 or v_norm < 1e-10 or w_norm < 1e-10:
                continue
            u = u / u_norm
            v = v / v_norm
            w = w / w_norm
            
            # Compute the residual
            residual = remaining.copy()
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[i,j,k] -= u[i*m+j] * v[j*p+k] * w[k*n+i]
            
            loss = np.sum(residual**2)
            if loss < best_loss:
                best_loss = loss
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        if best_u is None:
            break
        
        # Scale the rank-1 component to best fit
        # We want to minimize ||remaining - alpha * u ⊗ v ⊗ w||^2
        # alpha = <remaining, u⊗v⊗w> / ||u⊗v⊗w||^2
        numerator = 0.0
        denominator = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = best_u[i*m+j] * best_v[j*p+k] * best_w[k*n+i]
                    numerator += remaining[i,j,k] * prod
                    denominator += prod * prod
        
        if abs(denominator) < 1e-15:
            break
        alpha = numerator / denominator
        
        remaining -= alpha * np.einsum('a,b,c->abc', best_u.reshape(n,m), 
                                         best_v.reshape(m,p), best_w.reshape(p,n))
        
        U_list.append(alpha * best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        
        if np.linalg.norm(remaining) < 1e-10:
            break
    
    if len(U_list) == 0:
        return None
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    # Snap to half-integers
    U_snap = _snap_to_half_integers(U)
    V_snap = _snap_to_half_integers(V)
    W_snap = _snap_to_half_integers(W)
    
    ok, _ = _verify_decomposition(U_snap, V_snap, W_snap, n, m, p)
    if ok:
        return U_snap, V_snap, W_snap
    
    return None

def construct():
    n, m, p = DIMENSIONS
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Step 1: Compute matricization ranks to establish a lower bound
    lower_bound, (r1, r2, r3) = _compute_matricization_ranks(n, m, p)
    
    # The tensor rank is at least the maximum matricization rank
    # For matmul, we also know rank >= max(n, m, p) in many cases
    # But the matricization bound is rigorous
    search_start = max(lower_bound, 1)
    
    # Step 2: Try ranks from a reasonable upper bound down to the lower bound
    # Start from schoolbook rank and work down
    # But be smart: try a range around where we expect the answer to be
    
    # For 2x3x6, known results suggest rank might be around 24-30
    # Let's try from 30 down to the lower bound
    max_search_rank = min(30, n*m*p)
    
    for r in range(max_search_rank, search_start - 1, -1):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Try nonlinear optimization
        result = _try_rank_r(n, m, p, r, max_restarts=30, time_budget=TIME_LIMIT - 20)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break
        
        # Try direct construction for smaller ranks
        if r <= 20:
            result2 = _try_direct_construction(n, m, p, r)
            if result2 is not None:
                U, V, W = result2
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
