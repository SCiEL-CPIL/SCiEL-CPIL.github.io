import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def _verify_solution_int(U_int, V_int, W_int, n, m, p):
    """
    Verifies if U_int, V_int, W_int (integers, representing 2x values) 
    form a valid decomposition.
    Condition: sum_c (2*U) (2*V) (2*W) = 4 * delta
    => sum_c U_int V_int W_int = 4 * delta
    """
    rank = U_int.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    s = 0
                    for c in range(rank):
                        s += U_int[i * m + j, c] * V_int[jp * p + k, c] * W_int[k * n + i, c]
                    target = 4 if j == jp else 0
                    if s != target:
                        return False
    return True

def _snap_and_verify(U, V, W, n, m, p):
    """
    Snaps U, V, W to half-integers and verifies.
    Returns (U_list, V_list, W_list) if valid, else None.
    """
    # Snap to nearest half-integer
    # x -> round(2*x)/2
    U_half = np.round(U * 2) / 2.0
    V_half = np.round(V * 2) / 2.0
    W_half = np.round(W * 2) / 2.0
    
    # Convert to integers for exact verification
    U_int = np.round(U * 2).astype(np.int64)
    V_int = np.round(V * 2).astype(np.int64)
    W_int = np.round(W * 2).astype(np.int64)
    
    if _verify_solution_int(U_int, V_int, W_int, n, m, p):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
    # Try snapping to integers as well (in case half-integers were slightly off)
    U_int2 = np.round(U).astype(np.int64)
    V_int2 = np.round(V).astype(np.int64)
    W_int2 = np.round(W).astype(np.int64)
    
    if _verify_solution_int(U_int2, V_int2, W_int2, n, m, p):
        # If integers work, return them as floats (x.0)
        return U_int2.astype(float).tolist(), V_int2.astype(float).tolist(), W_int2.astype(float).tolist()
        
    return None

def _find_decomposition(rank, n, m, p, max_restarts=20, timeout=50):
    start_time = time.time()
    
    # Residual function for scipy.optimize.least_squares
    def residual_func(x):
        # x is flat array of length (n*m + m*p + p*n) * rank
        U_flat = x[:n*m*rank].reshape(n*m, rank)
        V_flat = x[n*m*rank:(n*m*rank + m*p*rank)].reshape(m*p, rank)
        W_flat = x[(n*m*rank + m*p*rank):].reshape(p*n, rank)
        
        residuals = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        # sum_c U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c]
                        # Use vectorized dot product for speed
                        prod = U_flat[i*m+j] * V_flat[jp*p+k] * W_flat[k*n+i]
                        val = np.dot(prod, np.ones(rank)) # Just sum
                        # Actually np.sum(prod) is faster
                        val = np.sum(prod)
                        target = 1.0 if j == jp else 0.0
                        residuals.append(val - target)
        return np.array(residuals)

    num_params = (n*m + m*p + p*n) * rank
    
    for r_idx in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Random initialization
        # Scale matters. If we start too large, gradients are huge.
        # If too small, they are tiny.
        # Standard deviation 0.1 is reasonable.
        x0 = np.random.randn(num_params) * 0.5
        
        try:
            result = least_squares(residual_func, x0, method='trf', 
                                   xtol=1e-12, ftol=1e-12, gtol=1e-12,
                                   max_nfev=1000)
            
            if result.success or result.cost < 1e-10:
                # Extract U, V, W
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
    # If we find a valid one, return it
    for r in range(19, 24):
        if time.time() > deadline - 10:
            break
            
        # Allocate more time for lower ranks? 
        # Or just fixed time per rank.
        # Total time available ~ 140s.
        # 5 ranks. 28s each?
        # Let's give 20s per rank.
        timeout = 20
        if time.time() + timeout > deadline - 5:
            timeout = deadline - time.time() - 5
            if timeout < 1:
                break
                
        result = _find_decomposition(r, n, m, p, max_restarts=50, timeout=timeout)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
