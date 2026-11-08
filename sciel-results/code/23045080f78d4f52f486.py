"""Search for rank-minimal exact tensor decomposition of 2x3 * 3x4 matmul."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]
DEADLINE = 140.0  # safety margin before 150s kill
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _is_half_integer(x, tol=1e-9):
    return abs(2*x - round(2*x)) < tol

def _check_decomposition(U, V, W, n=2, m=3, p=4, tol=1e-7):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Build via einsum: for each r, outer(U_col, V_col) gives (i,j,k) contributions weighted by W
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _check_half_integer_entries(U, V, W, tol=1e-6):
    """Check all entries are integer or half-integer."""
    for mat in (U, V, W):
        for row in mat:
            for val in row:
                if abs(2*val - round(2*val)) > tol:
                    return False
    return True

def _try_complete(U, V, n=2, m=3, p=4):
    """Given U and V (R columns), try to find W that completes the decomposition.
    Returns W if found, else None.
    
    For each column r, we need W[k*n+i, r] such that:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    This is a linear system in the W entries for each r? No, it's coupled across r.
    Actually, let's think of it differently:
    
    T[i,j,k] = sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    
    For fixed r, define A_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r]
    Then T = sum_r A_r * W_r where W_r[k,i] = W[k*n+i, r]
    
    This is a linear system: sum_r A_r ⊙ W_r (element-wise in (i,j,k)) = T
    where T[i,j,k] = 1.
    
    Unknowns: W[k*n+i, r] for all k,i,r -> p*n*R = 8*R unknowns
    Equations: n*m*p = 24 equations
    
    So we need R >= ceil(24/8) = 3? No, that's not right because the equations are nonlinear in U,V but linear in W given U,V.
    
    Given U,V, the system is linear in W:
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Let's index unknowns as W[k*n+i, r]. For equation (i,j,k), the unknowns that appear are W[k*n+i, r] for all r, with coefficient U[i*m+j, r] * V[j*p+k, r].
    
    So it's a linear system with 24 equations and 8R unknowns. We need 8R >= 24, i.e., R >= 3.
    But we also need the solution W to have half-integer entries.
    """
    R = U.shape[1]
    # Build linear system: A * w_vec = b
    # w_vec indexed by (k, i, r) -> index = k*n + i*R + r? Let's use index = ((k*n + i) * R + r)
    # Actually, W is stored as W[k*n+i, r], so let's flatten W to a vector of length p*n*R
    # Unknown: W[k*n+i, r], index in flat = (k*n+i)*R + r
    num_unknowns = p * n * R
    num_eqs = n * m * p
    
    A = np.zeros((num_eqs, num_unknowns))
    b = np.ones(num_eqs)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(R):
                    coeff = U[i*m + j, r] * V[j*p + k, r]
                    # Unknown W[k*n+i, r] is at flat index (k*n+i)*R + r
                    w_idx = (k*n + i) * R + r
                    A[eq_idx, w_idx] += coeff
                eq_idx += 1
    
    # Solve least squares / check consistency
    try:
        w_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
    except:
        return None
    
    # Check residual
    if len(residuals) > 0 and residuals[0] > 1e-8:
        return None
    
    # Reshape w_vec to W: W[k*n+i, r]
    W = w_vec.reshape(p*n, R)
    
    # Check if entries are (approximately) half-integers
    if not _check_half_integer_entries([U, V, W]):
        return None
    
    # Verify the decomposition
    if not _check_decomposition(U, V, W):
        return None
    
    return W

def _round_to_half(x, tol=1e-4):
    """Round to nearest half-integer if close enough."""
    rounded = round(2*x) / 2.0
    if abs(x - rounded) < tol:
        return rounded
    return None

def _try_round_and_verify(U, V, W):
    """Try rounding U, V, W to half-integers and verify."""
    U_r = np.zeros_like(U)
    V_r = np.zeros_like(V)
    W_r = np.zeros_like(W)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            rv = _round_to_half(U[i,j])
            if rv is None:
                return None
            U_r[i,j] = rv
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            rv = _round_to_half(V[i,j])
            if rv is None:
                return None
            V_r[i,j] = rv
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            rv = _round_to_half(W[i,j])
            if rv is None:
                return None
            W_r[i,j] = rv
    
    if _check_decomposition(U_r, V_r, W_r):
        return U_r, V_r, W_r
    return None

def _random_half_integer_matrix(rows, cols, max_val=4):
    """Generate random matrix with half-integer entries in [-max_val, max_val]."""
    M = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            # Choose from {-max_val, ..., max_val} in half-integer steps
            max_half = int(max_val * 2)
            val = random.randint(-max_half, max_half) / 2.0
            M[i,j] = val
    return M

def _search_rank(R, n=2, m=3, p=4, max_iters=500):
    """Search for a rank-R decomposition. Returns (U,V,W) if found, else None."""
    for it in range(max_iters):
        if _time_left() < 5:
            break
        # Random U and V with half-integer entries
        U = _random_half_integer_matrix(n*m, R, max_val=3)
        V = _random_half_integer_matrix(m*p, R, max_val=3)
        
        # Try to complete W
        W = _try_complete(U, V, n, m, p)
        if W is not None:
            # Try to round
            result = _try_round_and_verify(U, V, W)
            if result is not None:
                U_r, V_r, W_r = result
                # Convert to list of lists
                U_out = U_r.tolist()
                V_out = V_r.tolist()
                W_out = W_r.tolist()
                # Final verification
                if _check_decomposition(U_out, V_out, W_out):
                    return U_out, V_out, W_out
    return None

def _schoolbook(n=2, m=3, p=4):
    """Schoolbook decomposition of rank n*m*p."""
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

def construct():
    n, m, p = DIMENSIONS
    best_result = _schoolbook(n, m, p)
    
    # Try ranks from 3 upward (theoretical lower bound) up to 23
    # For each rank, do a limited number of random trials
    for R in range(3, 24):
        if _time_left() < 10:
            break
        # Allocate time budget per rank
        time_per_rank = min(5.0, _time_left() / max(1, 24 - R))
        max_iters = int(time_per_rank * 100)  # rough estimate
        max_iters = max(10, min(max_iters, 200))
        
        result = _search_rank(R, n, m, p, max_iters=max_iters)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            break  # Found minimum rank, stop
    
    return best_result
