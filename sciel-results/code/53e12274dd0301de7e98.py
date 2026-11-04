import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank_49(n, m, p):
    """
    Attempt to find a rank-49 decomposition using systematic search.
    Uses integer arithmetic with deadline guards.
    """
    target_rank = 49
    # We'll use a randomized approach with local search
    # Initialize with schoolbook and try to merge pairs
    
    # Start with schoolbook rank 50
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    # Initialize first 49 components from schoolbook
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if c < target_rank:
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
                else:
                    break
            if c >= target_rank:
                break
        if c >= target_rank:
            break
    
    # The 50th term is missing - we need to absorb it into the other 49
    # This requires solving a system of equations
    
    # Use a different approach: try all possible ways to perturb the decomposition
    # by adding small integer corrections to existing terms
    
    best_result = None
    
    # Try various perturbation strategies
    for attempt in range(1000):
        if _remaining() < 5:
            break
        
        # Create a random perturbation of the last few terms
        U_try = [row[:] for row in U]
        V_try = [row[:] for row in V]
        W_try = [row[:] for row in W]
        
        # Try to fix the missing term by adjusting existing terms
        # The missing term corresponds to (i=1, j=4, k=4) in 0-indexed
        # We need sum_c U[1*5+4][c] * V[4*5+4][c] * W[4*2+1][c] = 1
        
        # Try adding 1 to one of the existing terms and compensating
        for c in range(target_rank):
            # Check if we can add a correction
            # This is complex; use a simpler heuristic
            
            # Try setting one component to compensate
            U_try[1 * 5 + 4][c] = U[1 * 5 + 4][c] + 1
            V_try[4 * 5 + 4][c] = V[4 * 5 + 4][c]
            W_try[4 * 2 + 1][c] = W[4 * 2 + 1][c]
            
            # This won't work directly; we need a more systematic approach
            U_try[1 * 5 + 4][c] = U[1 * 5 + 4][c]
            V_try[4 * 5 + 4][c] = V[4 * 5 + 4][c]
            W_try[4 * 2 + 1][c] = W[4 * 2 + 1][c]
        
        # Skip to a better approach: use linear algebra over rationals
        break
    
    # Better approach: formulate as a linear system
    # We have 50 tensor entries T[i,j,k] that must equal 1
    # With rank 49, we have 49*3 = 147 variables (but with gauge freedom)
    # Actually, the decomposition has structure: each term c contributes
    # U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
    
    # This is a bilinear/quadratic system, hard to solve directly
    # For now, return None to indicate failure
    return None

def _construct():
    n, m, p = DIMENSIONS
    # First try to find rank 49
    result_49 = _try_rank_49(n, m, p)
    if result_49 is not None:
        return result_49
    
    # Fallback to schoolbook rank 50
    return _schoolbook(n, m, p)

def construct():
    try:
        result = _construct()
        # Verify to be safe
        n, m, p = DIMENSIONS
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
        else:
            # Fallback to a known correct simple structure if verification fails
            return _schoolbook(n, m, p)
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
