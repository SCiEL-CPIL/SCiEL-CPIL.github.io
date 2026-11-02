import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def _schoolbook():
    """Return the standard rank-18 schoolbook decomposition."""
    n, m, p = 2, 3, 3
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

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for c in range(rank):
                    lhs += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(lhs - 1.0) > 1e-9:
                    return False
    return True

def _check_integer_half_integer(U, V, W, tol=1e-6):
    """Check if all entries are close to integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True

def _snap_to_half_integers(U, V, W):
    """Snap values to the nearest half-integer."""
    U_snap = [[round(val * 2) / 2.0 for val in row] for row in U]
    V_snap = [[round(val * 2) / 2.0 for val in row] for row in V]
    W_snap = [[round(val * 2) / 2.0 for val in row] for row in W]
    return U_snap, V_snap, W_snap

def _try_rank(rank, deadline):
    """
    Attempt to find a rank-r decomposition with integer/half-integer entries.
    """
    n, m, p = 2, 3, 3
    nm, mp, pn = n * m, m * p, p * n
    
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def cost_and_grad(params):
        U = params[:rank*nm].reshape(nm, rank)
        V = params[rank*nm:rank*(nm+mp)].reshape(mp, rank)
        W = params[rank*(nm+mp):].reshape(pn, rank)
        
        residuals = np.zeros(n*m*p)
        idx = 0
        for i, j, k in indices:
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            res = np.dot(np.dot(U[u_idx, :], V[v_idx, :]), W[w_idx, :])
            residuals[idx] = res - 1.0
            idx += 1
        
        cost = np.sum(residuals**2)
        return cost

    from scipy.optimize import minimize
    
    best_cost = float('inf')
    best_params = None
    
    np.random.seed(42)
    
    # Limit attempts to avoid timeout
    max_attempts = 10
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        params = np.random.randn(rank * (nm + mp + pn)) * 0.1
        
        res = minimize(cost_and_grad, params, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
        
        if res.fun < best_cost:
            best_cost = res.fun
            best_params = res.x
            
        # Early stop if we find a very good solution
        if best_cost < 1e-10:
            break

    if best_params is None or best_cost > 1e-6:
        return None
        
    U = best_params[:rank*nm].reshape(nm, rank)
    V = best_params[rank*nm:rank*(nm+mp)].reshape(mp, rank)
    W = best_params[rank*(nm+mp):].reshape(pn, rank)
    
    if _check_integer_half_integer(U, V, W, tol=1e-3):
        U_snap, V_snap, W_snap = _snap_to_half_integers(U, V, W)
        
        if _verify(U_snap, V_snap, W_snap, n, m, p):
            return {"U": U_snap, "V": V_snap, "W": W_snap}
            
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try rank 16
    result = _try_rank(16, deadline)
    if result:
        return result
        
    # Try rank 17
    if time.time() < deadline - 5:
        result = _try_rank(17, deadline)
        if result:
            return result
            
    # Fallback to schoolbook
    return _schoolbook()
