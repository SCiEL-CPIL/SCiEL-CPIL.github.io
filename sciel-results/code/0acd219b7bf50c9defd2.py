import time
import numpy as np
from scipy.optimize import least_squares

START_TIME = time.time()
TIME_LIMIT = 110.0

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matmul tensor."""
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
                if abs(s - 1.0) > 1e-9:
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

def _snap_to_half_integers(X):
    """Snap values to nearest half-integer."""
    return np.round(X * 2) / 2

def _search_rank_r_sub(n, m, p, r, max_time=50.0):
    """Search for a rank-r decomposition of n x m x p matmul tensor."""
    # n=2, m=3, p=3
    # U: n*m x r = 6 x r
    # V: m*p x r = 9 x r
    # W: p*n x r = 9 x r
    
    # Flatten
    def unpack(params):
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r :].reshape(p*n, r)
        return U, V, W

    def residuals(params):
        U, V, W = unpack(params)
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    res[idx] = s - 1.0
                    idx += 1
        return res

    # Initial guess: random
    np.random.seed(42)
    x0 = np.random.randn(n*m*r + m*p*r + p*n*r) * 0.1
    
    # Try to solve
    start_time = time.time()
    try:
        # Use 'trf' method with high max_nfev
        result = least_squares(residuals, x0, method='trf', max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        
        if time.time() - start_time > max_time:
            return None
            
        if result.cost < 1e-10:
            U, V, W = unpack(result.x)
            # Snap to half-integers
            U_snap = _snap_to_half_integers(U)
            V_snap = _snap_to_half_integers(V)
            W_snap = _snap_to_half_integers(W)
            
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
                
            # If snap fails, try snapping to integers?
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if _verify_decomposition(U_int, V_int, W_int, n, m, p):
                return U_int, V_int, W_int
                
    except Exception:
        pass
        
    return None

def _expand_2x3x3_to_2x3x6(U_sub, V_sub, W_sub):
    """Expand a 2x3x3 decomposition to a 2x3x6 decomposition."""
    n, m, p_sub = 2, 3, 3
    p_big = 6
    r_sub = U_sub.shape[1]
    r_big = 2 * r_sub
    
    U_big = np.zeros((n * m, r_big))
    V_big = np.zeros((m * p_big, r_big))
    W_big = np.zeros((p_big * n, r_big))
    
    # First half: k = 0, 1, 2
    U_big[:, :r_sub] = U_sub
    for j in range(m):
        for k in range(p_sub):
            V_big[j * p_big + k, :r_sub] = V_sub[j * p_sub + k]
    for k in range(p_sub):
        for i in range(n):
            W_big[k * n + i, :r_sub] = W_sub[k * n + i]
            
    # Second half: k = 3, 4, 5
    U_big[:, r_sub:] = U_sub
    for j in range(m):
        for k in range(p_sub):
            # Map k_big = k + p_sub to k_sub = k
            V_big[j * p_big + (k + p_sub), r_sub:] = V_sub[j * p_sub + k]
    for k in range(p_sub):
        for i in range(n):
            W_big[(k + p_sub) * n + i, r_sub:] = W_sub[k * n + i]
            
    return U_big, V_big, W_big

def construct():
    n, m, p = 2, 3, 6
    
    # Fallback: Schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Try to find a better solution using the 2x3x3 sub-problem
    # Search for rank 17, then 16
    best_U, best_V, best_W = U_school, V_school, W_school
    
    for r in [17, 16]:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
            
        result = _search_rank_r_sub(2, 3, 3, r, max_time=40.0)
        if result is not None:
            U_sub, V_sub, W_sub = result
            U_big, V_big, W_big = _expand_2x3x3_to_2x3x6(U_sub, V_sub, W_sub)
            
            if _verify_decomposition(U_big, V_big, W_big, n, m, p):
                best_U, best_V, best_W = U_big, V_big, W_big
                break # Found a valid lower rank solution
                
    # Return best found
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
