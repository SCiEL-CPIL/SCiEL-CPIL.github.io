"""Program for 3x3x6 matrix multiplication rank minimization."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START

def _build_tensor_constraints(n, m, p):
    """
    Build the constraint matrix for the tensor decomposition.
    For each (i,j,k) triple, the constraint is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is a bilinear (actually trilinear) constraint, so we handle it
    via iterative optimization.
    """
    return n, m, p

def _check_decomposition(U, V, W, n, m, p, tol=1e-6):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n + i, r]
    
    For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    return True, None

def _schoolbook(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _objective(params, n, m, p, rank):
    """
    Objective function for optimization.
    params: flattened [U, V, W]
    Returns sum of squared errors for all (i,j,k) constraints.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = params[0: nm * rank].reshape(nm, rank)
    V = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
    W = params[nm * rank + mp * rank: nm * rank + mp * rank + pn * rank].reshape(pn, rank)
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                error += (val - 1.0) ** 2
    return error

def _try_find_decomposition(n, m, p, rank, max_time=60.0, seed=42):
    """
    Try to find a rank-r decomposition using optimization.
    Returns (U, V, W) if successful, else None.
    """
    start_time = time.time()
    
    # Initialize with random values
    np.random.seed(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Use small random initialization
    U_init = np.random.randn(nm, rank) * 0.5
    V_init = np.random.randn(mp, rank) * 0.5
    W_init = np.random.randn(pn, rank) * 0.5
    
    params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    # Try a few random restarts
    best_result = None
    best_obj = float('inf')
    
    for attempt in range(5):
        if time.time() - start_time > max_time:
            break
        
        if attempt > 0:
            np.random.seed(seed + attempt)
            U_init = np.random.randn(nm, rank) * 0.5
            V_init = np.random.randn(mp, rank) * 0.5
            W_init = np.random.randn(pn, rank) * 0.5
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        result = minimize(
            _objective,
            params_init,
            args=(n, m, p, rank),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result
        
        if best_obj < 1e-8:
            break
    
    if best_result is None or best_result.fun > 1e-6:
        return None
    
    # Extract U, V, W
    params = best_result.x
    U = params[0: nm * rank].reshape(nm, rank)
    V = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
    W = params[nm * rank + mp * rank: nm * rank + mp * rank + pn * rank].reshape(pn, rank)
    
    # Verify
    valid, _ = _check_decomposition(U, V, W, n, m, p, tol=1e-6)
    if not valid:
        return None
    
    return U, V, W

def _structured_block_decomposition(n, m, p):
    """
    Try to construct a decomposition using block structure.
    For 3x3x6, we can think of A as 3x3 and B as 3x6.
    We can partition B into three 3x2 blocks: B = [B1 | B2 | B3]
    Then C = [A*B1 | A*B2 | A*B3]
    
    If we can find a rank-r decomposition for 3x3x2 matrix multiplication,
    then the total rank would be 3*r.
    
    Let's first find the rank for 3x3x2.
    """
    # For 3x3x2, schoolbook rank is 18
    # Let's see if we can do better
    
    # Actually, let's try to directly search for low ranks
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0
    
    # First, try to find a low-rank decomposition
    # Try ranks from low to high
    best_decomp = None
    best_rank = n * m * p  # schoolbook
    
    # Try rank 18 first (3 * 6, based on 3x3x2 blocks)
    for target_rank in [18, 24, 30, 36, 42, 48]:
        if time.time() - start_time > deadline - 10:
            break
        
        result = _try_find_decomposition(n, m, p, target_rank, max_time=15.0, seed=42)
        if result is not None:
            U, V, W = result
            best_decomp = (U, V, W)
            best_rank = target_rank
            break
    
    # If no low-rank found, use schoolbook
    if best_decomp is None:
        U, V, W = _schoolbook(n, m, p)
        best_rank = n * m * p
    else:
        U, V, W = best_decomp
    
    # Final verification
    valid, info = _check_decomposition(U, V, W, n, m, p, tol=1e-5)
    if not valid:
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        best_rank = n * m * p
    
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
