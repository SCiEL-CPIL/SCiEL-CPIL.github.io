# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def _schoolbook(n, m, p):
    """Return U, V, W as 2D lists for schoolbook decomposition."""
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
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the tensor decomposition is exact."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[i * m + j, :], Va[j * p + k, :] * Wa[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _residuals(params, n, m, p, rank, fixed_mask):
    """
    params: flattened array of length rank * (nm + mp + pn)
    fixed_mask: boolean array of same length, True if variable is free, False if fixed.
    Returns residual vector of length n*m*p.
    """
    nm, mp, pn = n*m, m*p, p*n
    U = params[:rank*nm].reshape(nm, rank)
    V = params[rank*nm:rank*nm+rank*mp].reshape(mp, rank)
    W = params[rank*nm+rank*mp:].reshape(pn, rank)
    
    residuals = np.zeros(n*m*p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                residuals[idx] = s - 1.0
                idx += 1
    return residuals

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Try to reduce rank by 1, then 2, etc.
    # We start from schoolbook and try to zero out columns.
    # We solve for the free variables (all except the zeroed ones) to minimize residual.
    
    for target_rank in range(rank - 1, max(rank - 10, 0), -1):
        if time.time() > deadline:
            break
            
        # Try to zero out (rank - target_rank) columns.
        # We will try to zero out the last (rank - target_rank) columns.
        num_zero = rank - target_rank
        
        # Initialize params with schoolbook values
        params = np.concatenate([Ua.flatten(), Va.flatten(), Wa.flatten()])
        
        # Mask for free variables
        # We want to fix the last num_zero columns to 0.
        # The layout is: U (rank*nm), V (rank*mp), W (rank*pn)
        # Column c is at indices c*nm ... c*nm+nm-1 in U, etc.
        
        mask = np.ones(len(params), dtype=bool)
        
        # Zero out columns from rank-num_zero to rank-1
        for c in range(rank - num_zero, rank):
            # U part
            mask[c*nm : (c+1)*nm] = False
            # V part
            mask[rank*nm + c*mp : rank*nm + (c+1)*mp] = False
            # W part
            mask[rank*nm + rank*mp + c*pn : rank*nm + rank*mp + (c+1)*pn] = False
            
        # Extract free variables
        free_indices = np.where(mask)[0]
        x0 = params[free_indices]
        
        # Define the function to minimize
        def fun(x):
            params_new = params.copy()
            params_new[free_indices] = x
            return _residuals(params_new, n, m, p, rank, mask)
            
        try:
            # Use least_squares
            result = least_squares(fun, x0, method='lm', max_nfev=100)
            
            if result.success or np.min(result.cost) < 1e-6:
                # Reconstruct solution
                params_sol = params.copy()
                params_sol[free_indices] = result.x
                
                U_sol = params_sol[:rank*nm].reshape(nm, rank)
                V_sol = params_sol[rank*nm:rank*nm+rank*mp].reshape(mp, rank)
                W_sol = params_sol[rank*nm+rank*mp:].reshape(pn, rank)
                
                # Round to nearest integer/half-integer?
                # The problem requires integer/half-integer entries.
                # If the solution is close to integers, we can round.
                # Check if rounding preserves validity.
                
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                
                # If rounding fails, try to return the float solution?
                # The contract says "integer/half-integer entries".
                # So we MUST round.
                
        except Exception as e:
            pass
            
    # If no reduction found, return schoolbook
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # First, get schoolbook as fallback
    U, V, W = _schoolbook(n, m, p)
    
    # Try to reduce rank
    U, V, W = _try_reduce_rank(U, V, W, n, m, p, deadline)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
