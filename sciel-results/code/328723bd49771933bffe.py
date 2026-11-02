import time
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import differential_evolution

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
    Attempt to find a decomposition of the given rank using differential evolution.
    Returns (U, V, W) if found, else None.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    total_params = rank * (nm + mp + pn)
    
    start_time = time.time()
    
    # Define the objective function for differential_evolution
    def objective(x):
        U = x[:rank*nm].reshape(n, m, rank)
        V = x[rank*nm:rank*nm+rank*mp].reshape(m, p, rank)
        W = x[rank*nm+rank*mp:].reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        return np.sum((T - 1.0)**2)
    
    # Bounds: roughly -5 to 5 should cover half-integers up to 4.5
    bounds = [(-5, 5)] * total_params
    
    # We will try a few different strategies/restarts implicitly via DE population
    # DE is stochastic, so running it once with a decent population size might be enough,
    # or we run it multiple times if time permits.
    
    # Since DE can be slow, we limit the population size and generations.
    # For small rank (12-15), params are manageable (12*26=312 dims).
    # For larger rank, it gets harder.
    
    pop_size = 100
    max_iter = 500
    
    # We might need to tune these based on rank.
    # Let's just run DE with a timeout check inside if possible, 
    # but DE doesn't easily support callbacks for time.
    # So we rely on the outer loop to break if time runs out.
    
    try:
        # Use 'best1bin' mutation strategy which is standard
        # tol=1e-12, maxiter=100
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=max_iter, 
            popsize=10, # Keep small to fit in time
            tol=1e-12, 
            seed=42, # Fixed seed for reproducibility? No, random is better for search.
            # seed=None, 
            polish=False # We will polish manually to save time
        )
        
        if res.success or res.fun < 1e-10:
             # Check if we found a good solution
             x = res.x
             U = x[:rank*nm].reshape(n*m, rank)
             V = x[rank*nm:rank*nm+rank*mp].reshape(m*p, rank)
             W = x[rank*nm+rank*mp:].reshape(p*n, rank)
             
             # Snap
             U_snap = _snap_to_half_integers(U, tol=1e-5)
             V_snap = _snap_to_half_integers(V, tol=1e-5)
             W_snap = _snap_to_half_integers(W, tol=1e-5)
             
             if U_snap is not None and V_snap is not None and W_snap is not None:
                 if _validate_fast(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
                     return U_snap, V_snap, W_snap
    except Exception:
        pass
        
    return None

def _try_rank_ls(n, m, p, rank, time_budget=5.0, num_restarts=20):
    """
    Attempt to find a decomposition of the given rank using least squares with random restarts.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    total_params = rank * (nm + mp + pn)
    
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_budget:
            break
            
        scale = 1.0 / np.sqrt(rank)
        rng = np.random.RandomState(restart)
        params0 = rng.randn(total_params) * scale
        
        try:
            res = least_squares(
                _residuals, 
                params0, 
                args=(n, m, p, rank), 
                method='lm', 
                max_nfev=5000
            )
            
            if res.cost < 1e-12:
                U = res.x[:rank*nm].reshape(n*m, rank)
                V = res.x[rank*nm:rank*nm+rank*mp].reshape(m*p, rank)
                W = res.x[rank*nm+rank*mp:].reshape(p*n, rank)
                
                U_snap = _snap_to_half_integers(U, tol=1e-5)
                V_snap = _snap_to_half_integers(V, tol=1e-5)
                W_snap = _snap_to_half_integers(W, tol=1e-5)
                
                if U_snap is not None and V_snap is not None and W_snap is not None:
                    if _validate_fast(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
                        return U_snap, V_snap, W_snap
        except Exception:
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
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound up to schoolbook_rank - 1
    for rank in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        # Allocate time per rank
        # DE is slower per call than LM but potentially more robust.
        # Let's give DE 4s and LM 4s? Or just try DE first.
        # Let's try DE first with a short budget, then LM.
        
        # Try DE
        time_per_method = 4.0
        result = _try_rank(n, m, p, rank, time_budget=time_per_method)
        
        if result is None and time.time() < deadline - 5:
             # Try LM if DE failed or timed out
             result = _try_rank_ls(n, m, p, rank, time_budget=time_per_method)
             
        if result is not None:
            U, V, W = result
            best_rank = rank
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
