import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh_tridiagonal

def construct():
    """
    Optimizes the off-diagonal entries of a 12x12 tridiagonal matrix
    to minimize the variance of its eigenvalues (roots), which serves as
    a proxy for minimizing the uncertainty bound C4.
    """
    N = 12
    # Fixed diagonal entries based on Laguerre recurrence (2i+1 for i=0..11)
    # Note: roots_laguerre(12) corresponds to alpha=0.
    # Jacobi matrix for L_12:
    # d_i = 2i + 1
    # o_i = sqrt(i * (i+0)) = i for i=1..11? 
    # Actually, standard Jacobi for L_n:
    # a_k = 2k + alpha + 1
    # b_k = sqrt(k(k+alpha))
    # For alpha=0: a_k = 2k+1, b_k = k.
    # Indices 0 to 11.
    # d = [1, 3, 5, ..., 23]
    # o = [1, 2, ..., 11]
    
    d = np.array([2*i + 1 for i in range(N)], dtype=float)
    o_init = np.array([i for i in range(1, N)], dtype=float)
    
    def cost(params):
        o = params
        # Build matrix or use eigh_tridiagonal
        # eigh_tridiagonal takes d (diagonal) and e (off-diagonal)
        # e has length N-1
        try:
            w, _ = eigh_tridiagonal(d, o, select='all')
        except Exception:
            return 1e10
            
        if np.any(w <= 0):
            return 1e10 + np.sum(w[w <= 0])**2
            
        # Objective: Minimize variance
        var = np.var(w)
        return var

    # Initial guess: Standard Laguerre off-diagonals
    x0 = o_init.copy()
    
    # Bounds: Off-diagonals should be positive to maintain symmetry/positivity properties
    # Also, they shouldn't be 0 (would decouple).
    bounds = [(0.1, 100.0) for _ in range(N-1)]
    
    res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
    
    if res.success:
        o_opt = res.x
        w, _ = eigh_tridiagonal(d, o_opt, select='all')
        # Ensure positive
        if np.any(w <= 0):
            # Fallback to Laguerre roots if optimization failed positivity
            from scipy.special import roots_laguerre
            roots, _ = roots_laguerre(12)
            return [float(r) for r in roots]
        return [float(r) for r in w]
    else:
        # Fallback to Laguerre roots
        from scipy.special import roots_laguerre
        roots, _ = roots_laguerre(12)
        return [float(r) for r in roots]
