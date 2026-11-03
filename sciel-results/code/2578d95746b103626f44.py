"""Matrix multiplication 3x3 * 3x4 - rank-minimal exact tensor decomposition.

Uses structured random search with fast tensor evaluation, exploiting the
algebraic structure of matmul. Falls back to schoolbook rank-36 if no
better solution is found within the time budget.
"""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

# Time management
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # 110s deadline, 10s safety margin


def _time_left():
    return _DEADLINE - time.time()


def _schoolbook():
    """Rank-36 schoolbook construction."""
    n, m, p = DIMENSIONS
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


def _evaluate(U, V, W, n, m, p):
    """Compute the tensor from factors and return max error from 1.0.
    
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Should equal 1.0 for all valid i,j,k.
    """
    Ua = np.asarray(U, dtype=np.float64)
    Va = np.asarray(V, dtype=np.float64)
    Wa = np.asarray(W, dtype=np.float64)
    rank = Ua.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i]))
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _is_exact(U, V, W, n, m, p, tol=1e-9):
    return _evaluate(U, V, W, n, m, p) < tol


def _random_half_int(nrows, rank, scale=2):
    """Random half-integer matrix."""
    mat = []
    for _ in range(nrows):
        row = []
        for _ in range(rank):
            val = random.randint(-scale, scale) * 0.5
            row.append(val)
        mat.append(row)
    return mat


def _snap_to_half_int(arr):
    """Snap a numpy array to the nearest half-integer."""
    return np.round(arr * 2.0) / 2.0


def _objective_and_grad(x, n, m, p, rank):
    """Compute objective and gradient for flattened parameters.
    
    x is a vector of length 3 * rank * (n*m + m*p + p*n) ?? No, 
    U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    Total size = rank * (n*m + m*p + p*n).
    """
    # Unpack
    nm = n * m
    mp = m * p
    pn = p * n
    
    U_flat = x[:nm * rank]
    V_flat = x[nm * rank : nm * rank + mp * rank]
    W_flat = x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
    
    U = U_flat.reshape((nm, rank))
    V = V_flat.reshape((mp, rank))
    W = W_flat.reshape((pn, rank))
    
    loss = 0.0
    gU = np.zeros_like(U)
    gV = np.zeros_like(V)
    gW = np.zeros_like(W)
    
    # Precomputed indices
    # We can loop over i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                
                # s = sum_r U[ui, r] * V[vj, r] * W[wk, r]
                # Vectorized: np.dot(U[ui], V[vj] * W[wk])
                s = float(np.dot(U[ui], V[vj] * W[wk]))
                d = s - 1.0
                loss += d * d
                
                # Gradients
                # dL/dU[ui, r] = 2 * d * V[vj, r] * W[wk, r]
                # dL/dV[vj, r] = 2 * d * U[ui, r] * W[wk, r]
                # dL/dW[wk, r] = 2 * d * U[ui, r] * V[vj, r]
                
                gU[ui] += 2.0 * d * (V[vj] * W[wk])
                gV[vj] += 2.0 * d * (U[ui] * W[wk])
                gW[wk] += 2.0 * d * (U[ui] * V[vj])
    
    grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
    return loss, grad


def _try_rank_with_scipy(n, m, p, rank, seed, time_limit):
    """Try to find a decomposition of given rank using SciPy minimize."""
    rng = np.random.RandomState(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Initialize
    # Use a structured initialization: 
    # Each rank r corresponds to a "path". 
    # Simple init: random normal, scaled.
    # Better init: Try to make it close to schoolbook but compressed?
    # For now, random normal is a decent start for L-BFGS if we scale it right.
    
    U_init = rng.normal(0, 1.0, (nm, rank))
    V_init = rng.normal(0, 1.0, (mp, rank))
    W_init = rng.normal(0, 1.0, (pn, rank))
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    start_time = time.time()
    
    # Use L-BFGS-B
    # We don't have constraints on the continuous variables, so BFGS is good.
    # We need to be careful with the time limit.
    
    def obj_wrapper(x):
        loss, grad = _objective_and_grad(x, n, m, p, rank)
        return loss, grad
        
    # We can't easily interrupt L-BFGS-B mid-iteration in Python without a callback that raises an exception or checks time.
    # We will use a callback to check time.
    
    stop_flag = [False]
    def callback(xk):
        if time.time() > start_time + time_limit:
            stop_flag[0] = True
        return stop_flag[0]
        
    try:
        res = minimize(obj_wrapper, x0, method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'maxfun': 10000, 'callback': callback})
    except Exception:
        return None
        
    if stop_flag[0]:
        # If we hit time limit, we might still have a good result in res.x? 
        # Actually, if callback returns True, it stops. res.x is the last point.
        pass
        
    x_opt = res.x
    U_opt = x_opt[:nm * rank].reshape((nm, rank))
    V_opt = x_opt[nm * rank : nm * rank + mp * rank].reshape((mp, rank))
    W_opt = x_opt[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank].reshape((pn, rank))
    
    # Snap to half-integers
    U_snap = _snap_to_half_int(U_opt)
    V_snap = _snap_to_half_int(V_opt)
    W_snap = _snap_to_half_int(W_opt)
    
    # Verify
    U_list = U_snap.tolist()
    V_list = V_snap.tolist()
    W_list = W_snap.tolist()
    
    if _is_exact(U_list, V_list, W_list, n, m, p):
        return U_list, V_list, W_list
        
    return None


def _try_low_rank_search(n, m, p, deadline):
    """Search for low-rank decompositions starting from a reasonable upper bound."""
    # Schoolbook is 36.
    # Try ranks from 24 to 35.
    # We prioritize lower ranks.
    
    ranks_to_try = list(range(24, 36))
    # Maybe add 20-23 if we have time? Let's stick to 24-35 first to ensure we find *something*.
    # Actually, let's try 20..35.
    ranks_to_try = list(range(20, 36))
    
    # Shuffle slightly? No, lower is better.
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5.0:
            break
            
        # Try a few restarts for this rank
        # Give more time to lower ranks?
        # Let's say 5 restarts per rank, with 10s each? That's 50s.
        # We have ~110s total.
        # Let's allocate time dynamically.
        
        time_per_rank = min(10.0, (deadline - time.time()) / max(1, len(ranks_to_try)))
        
        for restart in range(5):
            if time.time() > deadline - 2.0:
                break
                
            time_limit_for_opt = time_per_rank / 5.0
            if time_limit_for_opt < 0.5:
                continue
                
            result = _try_rank_with_scipy(n, m, p, rank, seed=restart * 100 + rank, time_limit=time_limit_for_opt)
            if result is not None:
                return result
                
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matmul.
    
    Returns {'U', 'V', 'W'} with factor matrices of shapes:
    U: (n*m, rank) = (9, rank)
    V: (m*p, rank) = (12, rank)
    W: (p*n, rank) = (12, rank)
    
    Entries are integer or half-integer.
    """
    n, m, p = DIMENSIONS
    
    # Try structured search for low-rank solutions
    result = _try_low_rank_search(n, m, p, _DEADLINE)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank-36
    return _schoolbook()
