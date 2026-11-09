"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Fast vectorized verification"""
    R = len(U[0])
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m+j]
                v_val = V_arr[j*p+k]
                w_val = W_arr[k*n+i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _round_to_half(x):
    """Round to nearest half-integer"""
    return round(2*x) / 2.0

def _try_rank(R, n, m, p, max_attempts=200, time_limit=None):
    """Try to find a rank-R decomposition using structured + random search"""
    start = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    # Strategy 1: Try structured patterns based on known low-rank constructions
    # For 2x3x4, try to build from rank-7 2x2x2 blocks
    
    # Known rank-7 decomposition for 2x2x2 (Strassen):
    # We can embed this into our problem
    
    # Strategy: Use random initialization with half-integer values, then refine
    for attempt in range(max_attempts):
        if time_limit and time.time() - start > time_limit:
            break
        
        # Initialize with random half-integers in a reasonable range
        scale = 1.0
        U = [[_round_to_half(random.gauss(0, scale)) for _ in range(R)] for _ in range(nm)]
        V = [[_round_to_half(random.gauss(0, scale)) for _ in range(R)] for _ in range(mp)]
        W = [[_round_to_half(random.gauss(0, scale)) for _ in range(R)] for _ in range(pn)]
        
        # Check if it works
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W
        
        # Try iterative refinement: adjust individual entries
        for step in range(50):
            if time_limit and time.time() - start > time_limit:
                break
            
            # Pick random element to adjust
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, nm-1)
                col = random.randint(0, R-1)
                old = U[row][col]
                U[row][col] = _round_to_half(old + random.choice([-1, -0.5, 0.5, 1]))
                if _verify_fast(U, V, W, n, m, p):
                    return U, V, W
                else:
                    U[row][col] = old
            elif mat == 'V':
                row = random.randint(0, mp-1)
                col = random.randint(0, R-1)
                old = V[row][col]
                V[row][col] = _round_to_half(old + random.choice([-1, -0.5, 0.5, 1]))
                if _verify_fast(U, V, W, n, m, p):
                    return U, V, W
                else:
                    V[row][col] = old
            else:
                row = random.randint(0, pn-1)
                col = random.randint(0, R-1)
                old = W[row][col]
                W[row][col] = _round_to_half(old + random.choice([-1, -0.5, 0.5, 1]))
                if _verify_fast(U, V, W, n, m, p):
                    return U, V, W
                else:
                    W[row][col] = old
    
    # Strategy 2: Structured construction based on block decomposition
    # Try to use the rank-7 2x2x2 Strassen decomposition as a building block
    # Pad 2x3 to 3x3 and 3x4 to 4x4, or use other structures
    
    # For now, try a more systematic approach:
    # Fix W to have a specific structure and solve for U, V
    
    # Try: W[k*n+i, r] = delta_{k,0} * something + ...
    # This is getting complex. Let's try a different structured approach.
    
    # Strategy 3: Use the fact that for small ranks, we can enumerate
    # Try small integer values systematically
    if R <= 16:
        # Try all combinations of -1, 0, 1, 0.5, -0.5 for a subset
        # This is still too large for full enumeration, so use smart sampling
        pass
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p"""
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

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    Returns {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Try ranks from low to high
    for R in range(14, 25):
        if time.time() > deadline:
            break
        
        result = _try_rank(R, n, m, p, max_attempts=100, time_limit=deadline - time.time() - 5)
        if result is not None:
            U, V, W = result
            # Final verification
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
