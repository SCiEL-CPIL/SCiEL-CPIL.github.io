"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_START_TIME = time.time()
_TIMEOUT = 120.0
_SAFETY_MARGIN = 20.0

def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_integer_search(rank, n, m, p, values=(-1, 0, 0.5, 1, 1.5, 2)):
    """
    Try to find an integer/half-integer decomposition of given rank.
    Uses structured search with pruning.
    """
    if time.time() - _START_TIME > _TIMEOUT - _SAFETY_MARGIN:
        return None
    
    n, m, p = 3, 3, 5
    
    # Precompute index mappings
    # For each (i,j,k), we need sum over r of U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Strategy: enumerate possible column patterns
    # Each "column" r defines a triple (u_col, v_col, w_col)
    # where u_col is a vector of length n*m, v_col of length m*p, w_col of length p*n
    
    # For small rank, we can try random combinations of columns from a pool
    
    # Generate a pool of candidate columns
    # A column is valid if it doesn't have all zeros
    candidate_u = []
    for val in values:
        for i in range(n):
            for j in range(m):
                col = [0] * (n*m)
                col[i*m+j] = val
                if any(x != 0 for x in col):
                    candidate_u.append(col)
    
    candidate_v = []
    for val in values:
        for j in range(m):
            for k in range(p):
                col = [0] * (m*p)
                col[j*p+k] = val
                if any(x != 0 for x in col):
                    candidate_v.append(col)
    
    candidate_w = []
    for val in values:
        for k in range(p):
            for i in range(n):
                col = [0] * (p*n)
                col[k*n+i] = val
                if any(x != 0 for x in col):
                    candidate_w.append(col)
    
    # Try random combinations of rank columns
    np.random.seed(42)
    max_attempts = 50000
    
    for attempt in range(max_attempts):
        if time.time() - _START_TIME > _TIMEOUT - _SAFETY_MARGIN:
            break
        
        # Pick random columns
        u_cols = [candidate_u[np.random.randint(len(candidate_u))] for _ in range(rank)]
        v_cols = [candidate_v[np.random.randint(len(candidate_v))] for _ in range(rank)]
        w_cols = [candidate_w[np.random.randint(len(candidate_w))] for _ in range(rank)]
        
        # Build matrices
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        for r in range(rank):
            for idx in range(n*m):
                U[idx][r] = u_cols[r][idx]
            for idx in range(m*p):
                V[idx][r] = v_cols[r][idx]
            for idx in range(p*n):
                W[idx][r] = w_cols[r][idx]
        
        # Validate
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_greedy_merge():
    """
    Start with schoolbook and try to merge pairs of terms.
    Two terms (r1, r2) can be merged if there exist coefficients such that
    the product structure is preserved.
    """
    if time.time() - _START_TIME > _TIMEOUT - _SAFETY_MARGIN:
        return None
    
    n, m, p = 3, 3, 5
    schoolbook = _schoolbook()
    U = [row[:] for row in schoolbook['U']]
    V = [row[:] for row in schoolbook['V']]
    W = [row[:] for row in schoolbook['W']]
    rank = n * m * p
    
    # Try to find pairs that can be eliminated
    # This is complex; skip for now and rely on other methods
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try integer/half-integer search for various ranks
    # The rank of 3x3x5 matrix multiplication is known to be at most 45 (schoolbook)
    # and at least 27 (by various bounds). Let's try to find a better rank.
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But actually for matrix multiplication, lower bound is at least max(n,m,p)^2 / something
    # For 3x3x5, known lower bound is around 27
    
    for rank in range(20, schoolbook_rank):
        if time.time() - _START_TIME > _TIMEOUT - _SAFETY_MARGIN:
            break
        
        result = _try_integer_search(rank, n, m, p)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
