import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def _validate_fast(U, V, W, n, m, p, tol=1e-10):
    """Fast vectorized validation using einsum."""
    rank = U.shape[1]
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.max(np.abs(T - 1.0)) < tol

def _snap_to_half_integers(arr, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    # Multiply by 2, round, divide by 2
    candidates = np.round(arr * 2) / 2
    if np.max(np.abs(arr - candidates)) < tol:
        return candidates
    return None

def _residuals(params, n, m, p, rank):
    """
    Residuals for least_squares.
    params: flat array of length rank * (n*m + m*p + p*n)
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = params[:rank*nm].reshape(n, m, rank)
    V = params[rank*nm:rank*nm+rank*mp].reshape(m, p, rank)
    W = params[rank*nm+rank*mp:].reshape(p, n, rank)
    
    # Compute T[i,j,k]
    T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
    
    # Residuals are T - 1
    return (T - 1.0).flatten()

def _try_rank(n, m, p, rank, time_budget=5.0, num_restarts=20):
    """
    Attempt to find a decomposition of the given rank.
    Returns (U, V, W) if found, else None.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    total_params = rank * (nm + mp + pn)
    
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_budget:
            break
            
        # Initialize with small random values
        # Scale matters. If we use 1.0, gradients might be large.
        # Let's try scaling by 1/sqrt(rank) or similar.
        scale = 1.0 / np.sqrt(rank)
        rng = np.random.RandomState(restart)
        params0 = rng.randn(total_params) * scale
        
        try:
            res = least_squares(
                _residuals, 
                params0, 
                args=(n, m, p, rank), 
                method='lm', 
                max_nfev=10000
            )
            
            if res.cost < 1e-12:
                # Extract U, V, W
                U = res.x[:rank*nm].reshape(n*m, rank)
                V = res.x[rank*nm:rank*nm+rank*mp].reshape(m*p, rank)
                W = res.x[rank*nm+rank*mp:].reshape(p*n, rank)
                
                # Try to snap to half-integers
                U_snap = _snap_to_half_integers(U, tol=1e-5)
                V_snap = _snap_to_half_integers(V, tol=1e-5)
                W_snap = _snap_to_half_integers(W, tol=1e-5)
                
                if U_snap is not None and V_snap is not None and W_snap is not None:
                    if _validate_fast(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
                        return U_snap, V_snap, W_snap
                else:
                    # If snapping fails, check if it's valid as float? 
                    # The prompt requires integer/half-integer. 
                    # But maybe the "exact" decomposition implies we can return the floats 
                    # if they are extremely close? No, "integer/half-integer entries".
                    # So we must snap.
                    pass
                    
        except Exception as e:
            continue
            
    return None

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 2x3 by 3x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook as default
    U, V, W = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    best_result = (U, V, W)
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound up to schoolbook_rank - 1
    # We prioritize lower ranks.
    for rank in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 2:
            break
        
        # Allocate time per rank
        # Lower ranks are harder, give them more time?
        # Or just fixed time per rank.
        # There are 12 ranks (12 to 23).
        # 140s / 12 ranks ~ 11s per rank.
        # Let's give 10s per rank.
        time_per_rank = 10.0
        
        result = _try_rank(n, m, p, rank, time_budget=time_per_rank, num_restarts=50)
        
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_result = (U, V, W)
            # If we found a very low rank, we might want to stop?
            # But we want the MINIMAL. So we continue to check lower ranks.
            # However, we are iterating upwards from lower_bound.
            # So the first one we find is the minimal!
            # Wait, the loop is `for rank in range(lower_bound, schoolbook_rank)`.
            # So if we find rank 12, we should return it immediately?
            # Yes, because we are looking for rank-minimal.
            # But what if rank 12 fails, and rank 13 succeeds? Then 13 is the minimal.
            # So we break as soon as we find a valid one.
            break
    
    # Final validation
    if not _validate_fast(U, V, W, n, m, p, tol=1e-10):
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
        best_rank = schoolbook_rank

    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
