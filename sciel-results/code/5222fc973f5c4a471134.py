import numpy as np
import time
from scipy.optimize import least_squares
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def _verify_solution_int(U_int, V_int, W_int, n, m, p, target):
    """
    Verifies if U_int, V_int, W_int (integers) form a valid decomposition.
    Condition: sum_c U_int V_int W_int = target * delta
    """
    rank = U_int.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    s = 0
                    for c in range(rank):
                        s += U_int[i * m + j, c] * V_int[jp * p + k, c] * W_int[k * n + i, c]
                    t = target if j == jp else 0
                    if s != t:
                        return False
    return True

def _snap_and_verify(U, V, W, n, m, p):
    """
    Snaps U, V, W to half-integers or integers and verifies.
    Returns (U_list, V_list, W_list) if valid, else None.
    """
    
    # Try half-integers first (more likely for optimal rank)
    # U = u/2 => 2U = u (int)
    # Sum (u/2)(v/2)(w/2) = 1 => Sum uvw = 8
    U_int = np.round(U * 2).astype(np.int64)
    V_int = np.round(V * 2).astype(np.int64)
    W_int = np.round(W * 2).astype(np.int64)
    
    if _verify_solution_int(U_int, V_int, W_int, n, m, p, target=8):
        U_half = U_int / 2.0
        V_half = V_int / 2.0
        W_half = W_int / 2.0
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
    # Try integers
    # U = u (int)
    # Sum u v w = 1
    U_int2 = np.round(U).astype(np.int64)
    V_int2 = np.round(V).astype(np.int64)
    W_int2 = np.round(W).astype(np.int64)
    
    if _verify_solution_int(U_int2, V_int2, W_int2, n, m, p, target=1):
        return U_int2.astype(float).tolist(), V_int2.astype(float).tolist(), W_int2.astype(float).tolist()
        
    return None

def _find_decomposition(rank, n, m, p, max_restarts=50, timeout=50):
    start_time = time.time()
    
    def residual_func(x):
        U_flat = x[:n*m*rank].reshape(n*m, rank)
        V_flat = x[n*m*rank:(n*m*rank + m*p*rank)].reshape(m*p, rank)
        W_flat = x[(n*m*rank + m*p*rank):].reshape(p*n, rank)
        
        residuals = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        prod = U_flat[i*m+j] * V_flat[jp*p+k] * W_flat[k*n+i]
                        val = np.sum(prod)
                        target = 1.0 if j == jp else 0.0
                        residuals.append(val - target)
        return np.array(residuals)

    num_params = (n*m + m*p + p*n) * rank 
    best_cost = np.inf
    best_x = None

    for r_idx in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        x0 = np.random.randn(num_params) * 0.5
        
        try:
            # Phase 1: Least Squares
            result = least_squares(residual_func, x0, method='trf', 
                                   xtol=1e-10, ftol=1e-10, gtol=1e-10,
                                   max_nfev=2000)
            
            if result.cost < 1e-8:
                best_x = result.x
                best_cost = result.cost
                
                # Phase 2: Polish with minimize (L-BFGS-B) to get closer to 0
                # We minimize sum(residuals^2)
                def obj(x):
                    res = residual_func(x)
                    return np.sum(res**2)
                
                def grad(x):
                    # Gradient of sum(res^2) is 2 * J^T * res
                    # least_squares provides jac? No, we can use finite diff or analytic.
                    # For speed, just use obj. L-BFGS-B uses finite diff if grad not provided.
                    # But for high dim, this is slow.
                    # Let's stick to least_squares result if cost is low enough.
                    pass
                
                # If cost is very low, try to snap immediately
                U = result.x[:n*m*rank].reshape(n*m, rank)
                V = result.x[n*m*rank:(n*m*rank + m*p*rank)].reshape(m*p, rank)
                W = result.x[(n*m*rank + m*p*rank):].reshape(p*n, rank)
                
                snapped = _snap_and_verify(U, V, W, n, m, p)
                if snapped is not None:
                    return snapped
                    
        except Exception:
            continue
            
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = 140  # Leave 10s margin for overhead and return
    
    # Try ranks from 19 up to 23
    for r in range(19, 24):
        if time.time() > deadline - 10:
            break
            
        # Allocate time based on remaining budget
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
            
        # Give more time to lower ranks as they are harder
        # 19: 40s, 20: 30s, 21: 20s, 22: 15s, 23: 10s
        timeout_map = {19: 40, 20: 30, 21: 20, 22: 15, 23: 10}
        timeout = min(timeout_map.get(r, 10), remaining)
        
        result = _find_decomposition(r, n, m, p, max_restarts=100, timeout=timeout)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    return _schoolbook()

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
