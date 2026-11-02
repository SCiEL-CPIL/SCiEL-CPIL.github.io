"""Matrix multiplication 2x5x5 rank-minimal exact tensor decomposition search."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# Known upper bound: schoolbook rank = 2*5*5 = 50
def _schoolbook():
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

def _verify(U, V, W, n=2, m=5, p=5, tol=1e-9):
    """Verify that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all i,j,k."""
    r = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _to_int_or_half(values):
    """Check if all values are integer or half-integer, return as list of Fractions if so."""
    result = []
    for v in values:
        f = Fraction(v).limit_denominator(2)
        if abs(f - v) > 1e-12:
            return None
        result.append(f)
    return result

def _try_rank(r, deadline, rng_seed=42):
    """Try to find a rank-r decomposition. Return dict or None."""
    n, m, p = DIMENSIONS
    import numpy as np
    
    # Strategy: random search with structured initialization
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need U (nm x r), V (mp x r), W (pn x r) such that
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For small ranks, use random initialization with values in {-1, 0, 1, 2}
    # and try to verify.
    
    max_attempts = 2000
    vals = [-2, -1, 0, 1, 2]
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        np.random.seed(rng_seed + attempt)
        
        # Generate random U, V, W with small integer entries
        U = np.random.choice(vals, size=(n*m, r))
        V = np.random.choice(vals, size=(m*p, r))
        W = np.random.choice(vals, size=(p*n, r))
        
        # Quick check: verify decomposition
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if s != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s to leave margin
    
    # Start from known best (schoolbook = 50)
    best = _schoolbook()
    best_rank = 50
    
    # Try to improve with smaller ranks
    # Known theoretical lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # Actually for matrix multiplication, the rank must be at least max(n*m, m*p, p*n)
    # But there might be better bounds. Let's try ranks from 25 upward.
    
    # First try some structured approaches for specific ranks
    
    # Approach 1: For rank r where r >= m=5, we can try to factor as:
    # U[i,j,c], V[j,k,c], W[k,i,c]
    # One structure: W depends only on k and i, U on i and j, V on j and k
    
    # Try rank = 5, 6, 7, ... up to 49
    for r in range(5, best_rank):
        if time.time() > deadline - 10:
            break
        result = _try_rank(r, deadline)
        if result is not None:
            best = result
            best_rank = r
            break
    
    # Also try a few more specific structured constructions
    # For 2x5x5, one known approach:
    # Split the 5x5 multiplication into blocks
    
    # Try rank 25 (which is m*p = 25, also p*n... wait n=2,p=5 so p*n=10)
    # Actually the lower bound is max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    
    return best

# Ensure the function is defined at module level
def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # Default: schoolbook rank 50
    best = _schoolbook()
    best_rank = 50
    
    # Try randomized search for smaller ranks
    for r in range(5, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(r, deadline)
        if result is not None:
            best = result
            best_rank = r
            break
    
    return best
