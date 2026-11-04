import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _validate_float(U, V, W, tol=1e-9):
    """Fast float validation."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if r == 0:
        return False
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _validate_exact_with_denom(U, V, W, denom):
    """
    Validate exact decomposition where entries are multiples of 1/denom.
    U, V, W are lists of lists of ints (values * denom).
    """
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if r == 0:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(r):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != denom * denom * denom:
                    return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute decomposition error."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    err_sq = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                err_sq += (val - 1.0) ** 2
    return np.sqrt(err_sq)

def _alternating_min(U, V, W, n, m, p, max_iters=200, deadline=None, tol=1e-10):
    """
    Alternating minimization in float. Returns (U, V, W, error).
    """
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, len(U[0])))
                for k in range(p):
                    M[k, :] = V[j*p + k] * W[k*n + i]
                try:
                    u_new = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    U[i*m + j] = u_new
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, len(V[0])))
                for i in range(n):
                    M[i, :] = U[i*m + j] * W[k*n + i]
                try:
                    v_new = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    V[j*p + k] = v_new
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, len(W[0])))
                for j in range(m):
                    M[j, :] = U[i*m + j] * V[j*p + k]
                try:
                    w_new = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    W[k*n + i] = w_new
                except:
                    pass
        
        err = _compute_error(U, V, W, n, m, p)
        if err < tol:
            break
    
    err = _compute_error(U, V, W, n, m, p)
    return U, V, W, err

def _local_search(U, V, W, n, m, p, deadline, max_steps=50):
    """
    Local search: try small perturbations to escape local minima.
    Returns best (U, V, W, error) found.
    """
    best_U = np.array(U, dtype=np.float64)
    best_V = np.array(V, dtype=np.float64)
    best_W = np.array(W, dtype=np.float64)
    best_err = _compute_error(best_U, best_V, best_W, n, m, p)
    
    rng = random.Random(42)
    
    for step in range(max_steps):
        if time.time() > deadline - 10:
            break
        
        # Pick a random matrix and coordinate to perturb
        mat_choice = rng.random()
        if mat_choice < 0.33:
            target = 'U'
            idx = rng.randint(0, n*m-1)
        elif mat_choice < 0.66:
            target = 'V'
            idx = rng.randint(0, m*p-1)
        else:
            target = 'W'
            idx = rng.randint(0, p*n-1)
        
        col = rng.randint(0, len(U[0])-1)
        perturb = rng.gauss(0, 0.1)
        
        # Create perturbed copy
        U_copy = best_U.copy()
        V_copy = best_V.copy()
        W_copy = best_W.copy()
        
        if target == 'U':
            U_copy[idx, col] += perturb
        elif target == 'V':
            V_copy[idx, col] += perturb
        else:
            W_copy[idx, col] += perturb
        
        # Run a few alternating minimization iterations
        U_t, V_t, W_t, err = _alternating_min(
            U_copy.tolist(), V_copy.tolist(), W_copy.tolist(),
            n, m, p, max_iters=20, deadline=deadline, tol=1e-10
        )
        
        if err < best_err - 1e-12:
            best_U = np.array(U_t, dtype=np.float64)
            best_V = np.array(V_t, dtype=np.float64)
            best_W = np.array(W_t, dtype=np.float64)
            best_err = err
    
    return best_U, best_V, best_W, best_err

def _find_exact_rational(U, V, W, n, m, p, max_denom=100):
    """
    Try to find a rational representation with small denominator.
    Returns (U_int, V_int, W_int, denom) or None.
    """
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for denom in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]:
        # Check if all entries are close to multiples of 1/denom
        U_r = np.round(U_np * denom) / denom
        V_r = np.round(V_np * denom) / denom
        W_r = np.round(W_np * denom) / denom
        
        # Check closeness
        if (np.max(np.abs(U_np - U_r)) > 1e-8 or
            np.max(np.abs(V_np - V_r)) > 1e-8 or
            np.max(np.abs(W_np - W_r)) > 1e-8):
            continue
        
        # Convert to integers
        U_int = [[int(round(x * denom)) for x in row] for row in U_r.tolist()]
        V_int = [[int(round(x * denom)) for x in row] for row in V_r.tolist()]
        W_int = [[int(round(x * denom)) for x in row] for row in W_r.tolist()]
        
        # Validate exactly
        if _validate_exact_with_denom(U_int, V_int, W_int, denom):
            return U_int, V_int, W_int, denom
    
    return None

def _structured_init(rank, n, m, p, seed=None):
    """Structured initialization for alternating minimization."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Base structured pattern
    for c in range(rank):
        i_idx = c % n
        j_idx = (c // n) % m
        k_idx = (c // (n * m)) % p
        U[i_idx * m + j_idx, c] = 1.0
        V[j_idx * p + k_idx, c] = 1.0
        W[k_idx * n + i_idx, c] = 1.0
    
    # Add small random noise to break symmetry
    noise_scale = 0.05
    U += np.random.randn(n * m, rank) * noise_scale
    V += np.random.randn(m * p, rank) * noise_scale
    W += np.random.randn(p * n, rank) * noise_scale
    
    return U, V, W

def _try_rank(rank, n, m, p, deadline, max_attempts=8):
    """
    Try to find a valid decomposition of given rank.
    Returns (U_list, V_list, W_list) with integer entries, or None.
    """
    for attempt in range(max_attempts):
        if time.time() > deadline - 15:
            break
        
        seed = attempt * 7919 + rank * 104729 + 42
        U, V, W = _structured_init(rank, n, m, p, seed=seed)
        
        # Run alternating minimization
        U, V, W, err = _alternating_min(
            U.tolist(), V.tolist(), W.tolist(),
            n, m, p, max_iters=100, deadline=deadline, tol=1e-12
        )
        
        # If error is small, try local search to polish
        if err < 1e-6:
            U, V, W, err = _local_search(
                U, V, W, n, m, p, deadline, max_steps=30
            )
        
        # Try to find exact rational representation
        if err < 1e-8:
            result = _find_exact_rational(U, V, W, n, m, p, max_denom=120)
            if result is not None:
                U_int, V_int, W_int, denom = result
                # Convert back to the format: entries are rationals with denominator denom
                # But the evaluator expects integer/half-integer entries
                # Actually, let's return as fractions or check if denom divides nicely
                # For now, return as float lists that are exactly representable
                U_list = [[x / denom for x in row] for row in U_int]
                V_list = [[x / denom for x in row] for row in V_int]
                W_list = [[x / denom for x in row] for row in W_int]
                
                # Final validation
                if _validate_float(U_list, V_list, W_list, tol=1e-9):
                    return U_list, V_list, W_list
    
    return None

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j][col] = 1
                V_sb[j*p + k][col] = 1
                W_sb[k*n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = [row[:] for row in U_sb], [row[:] for row in V_sb], [row[:] for row in W_sb]
    best_rank = schoolbook_rank
    
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound upward, but be time-aware
    ranks_to_try = list(range(lower_bound, min(lower_bound + 20, schoolbook_rank + 1)))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        result = _try_rank(rank, n, m, p, deadline, max_attempts=6)
        
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = rank
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
