"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact using numpy for speed."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _snap_to_half_int(arr):
    """Snap values to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _solve_linear_for_factor(fixed_factors, target_tensor, n, m, p, factor_idx, rank):
    """
    Solve for one factor (U, V, or W) given the other two are fixed.
    factor_idx: 0 for U, 1 for V, 2 for W.
    Returns the solved factor as a numpy array.
    """
    # We construct the linear system A x = b
    # The tensor equation is sum_r U_{u,r} V_{v,r} W_{w,r} = T_{u,v,w}
    # We flatten the indices (u, v, w) to a single index for the vector of equations.
    
    U, V, W = fixed_factors
    
    # Number of equations
    num_eq = n * m * p
    
    # Number of unknowns
    if factor_idx == 0: # U
        num_unknowns = n * m * rank
    elif factor_idx == 1: # V
        num_unknowns = m * p * rank
    else: # W
        num_unknowns = p * n * rank
        
    A = np.zeros((num_eq, num_unknowns))
    b = np.zeros(num_eq)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                b[eq_idx] = 1.0 # Target tensor entry is 1
                
                if factor_idx == 0: # Solve for U
                    # Coefficients for U_{u, r} are V_{v, r} * W_{w, r}
                    for r in range(rank):
                        A[eq_idx, u_idx * rank + r] = V[v_idx, r] * W[w_idx, r]
                elif factor_idx == 1: # Solve for V
                    # Coefficients for V_{v, r} are U_{u, r} * W_{w, r}
                    for r in range(rank):
                        A[eq_idx, v_idx * rank + r] = U[u_idx, r] * W[w_idx, r]
                else: # Solve for W
                    # Coefficients for W_{w, r} are U_{u, r} * V_{v, r}
                    for r in range(rank):
                        A[eq_idx, w_idx * rank + r] = U[u_idx, r] * V[v_idx, r]
                
                eq_idx += 1
                
    # Solve least squares
    x, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
    
    # Reshape
    if factor_idx == 0:
        return x.reshape(n * m, rank)
    elif factor_idx == 1:
        return x.reshape(m * p, rank)
    else:
        return x.reshape(p * n, rank)

def _als_decomposition(n, m, p, rank, max_iters=50, seed=42):
    """
    Alternating Least Squares with snapping to find a decomposition.
    """
    np.random.seed(seed)
    
    # Initialize with random integers
    U = np.random.randint(-2, 3, (n * m, rank))
    V = np.random.randint(-2, 3, (m * p, rank))
    W = np.random.randint(-2, 3, (p * n, rank))
    
    best_rank_found = None
    best_factors = None
    
    for it in range(max_iters):
        # Solve for U
        U_sol = _solve_linear_for_factor((U, V, W), None, n, m, p, 0, rank)
        U = _snap_to_half_int(U_sol)
        
        # Solve for V
        V_sol = _solve_linear_for_factor((U, V, W), None, n, m, p, 1, rank)
        V = _snap_to_half_int(V_sol)
        
        # Solve for W
        W_sol = _solve_linear_for_factor((U, V, W), None, n, m, p, 2, rank)
        W = _snap_to_half_int(W_sol)
        
        # Check if exact
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        # Optional: Check if we got stuck (same factors)
        # For now, just continue
        
    return None

def construct():
    start_time = time.time()
    deadline = 145
    
    n, m, p = DIMENSIONS
    
    # Try ranks from 55 down to 50
    # 55 is known to be possible (from inspiration)
    # We will try to find a solution with rank 55 first, then try lower.
    
    # Fallback
    fallback = _schoolbook(n, m, p)
    
    # Try ALS for different ranks
    for rank in [55, 54, 53, 52, 51, 50]:
        if time.time() > deadline:
            break
            
        # Try multiple seeds for robustness
        for seed in range(10):
            if time.time() > deadline:
                break
                
            res = _als_decomposition(n, m, p, rank, max_iters=20, seed=seed)
            if res is not None:
                return res
                
    return fallback
# EVOLVE-BLOCK-END
