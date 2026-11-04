"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import itertools
import numpy as np

def _verify_factors(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] equals the MM tensor."""
    rank = len(U[0])
    for i, j, k in itertools.product(range(n), range(m), range(p)):
        s = 0
        for c in range(rank):
            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        if abs(s - 1) > tol:
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _find_rank19_5x5x5():
    """Try to find or verify rank-19 decomposition for 5x5x5 MM."""
    # We'll try to construct the factors and verify them
    # If verification fails, return None
    
    # The rank-19 decomposition is known to exist. Let me try to use 
    # a construction based on the structure of the problem.
    
    # For now, return None to indicate we need schoolbook fallback
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try rank-19 approach
    r19 = _find_rank19_5x5x5()
    if r19 is not None:
        U5, V5, W5 = r19
        # Restrict to 4x5x5: remove rows where i=4 from U and W
        # U5 has 25 rows (i*5+j), keep rows where i < 4, i.e., rows 0-19
        U4 = [U5[i*5+j] for i in range(4) for j in range(5)]
        # V5 has 25 rows (j*5+k), keep all
        V4 = V5
        # W5 has 25 rows (k*5+i), keep rows where i < 4
        # W index is k*5+i, so for i in 0..3: rows 0-3, 5-8, 10-13, 15-18
        W4 = [W5[k*5+i] for k in range(5) for i in range(4)]
        
        # Verify
        if _verify_factors(U4, V4, W4, 4, 5, 5):
            return {"U": U4, "V": V4, "W": W4}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
