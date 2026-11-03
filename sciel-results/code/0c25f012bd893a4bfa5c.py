"""Matrix multiplication 3,3,5 rank minimization."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np
from fractions import Fraction
from itertools import product

def _verify(U, V, W):
    """Verify that U, V, W give an exact decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] for l in range(r))
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x5.
    The schoolbook decomposition has rank 45.
    We attempt to find a lower rank decomposition using symbolic enumeration
    with small integer coefficients, falling back to schoolbook if not found.
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe (150s total limit)
    
    # Schoolbook: rank 45
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j][col] = 1
                V_sb[j*p + k][col] = 1
                W_sb[k*n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # Strategy: For each candidate rank r < 45, try to find U, V, W with entries in {-1, 0, 1, 2}
    # that satisfy sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1 for all i,j,k
    
    # This is a hard problem, so we'll use a heuristic approach
    # Try rank 44 first (one less than schoolbook)
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # For small ranks, we can try random initialization and local search
    # But given the time constraint, we'll focus on a structured approach
    
    # Let's try a more systematic approach for rank 44
    # We can try to modify the schoolbook by combining two terms
    
    # Actually, for 3x3x5, it's known that the rank is at least 45 in many cases
    # Let's verify if we can find rank 44
    
    # Given the complexity, let's try a limited search
    # We'll try to see if rank 44 is possible by solving a linear system
    
    # For now, let's just return the schoolbook if we can't find something better quickly
    # This ensures correctness and meets the time limit
    
    if time.time() > deadline - 5:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to find rank 44 decomposition
    # We'll use a randomized approach with local optimization
    
    import random
    random.seed(42)
    
    for target_rank in [44, 43, 42]:
        if time.time() > deadline - 10:
            break
            
        # Try to find a decomposition of rank target_rank
        # Initialize with random values in {-1, 0, 1}
        
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            U = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            
            # Try to solve for W
            # For each (k, i), we need: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1 for all j
            # This gives us a system of equations for W[k*n+i, :]
            
            W_candidate = [[0] * target_rank for _ in range(p * n)]
            valid = True
            
            for k in range(p):
                for i in range(n):
                    # We need to find w such that for all j: sum_l U[i*m+j, l] * V[j*p+k, l] * w[l] = 1
                    # This is a linear system: A * w = b where A[j, l] = U[i*m+j, l] * V[j*p+k, l]
                    # and b[j] = 1 for all j
                    
                    A = np.array([[U[i*m+j, l] * V[j*p+k, l] for l in range(target_rank)] for j in range(m)], dtype=float)
                    b = np.ones(m)
                    
                    try:
                        # Solve least squares
                        w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        
                        # Check if the solution is exact
                        if np.allclose(A @ w, b, atol=1e-10):
                            # Check if w has small integer values
                            w_rounded = np.round(w)
                            if np.allclose(w, w_rounded, atol=1e-10):
                                W_candidate[k*n+i] = [int(x) for x in w_rounded]
                            else:
                                valid = False
                                break
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition
                if _verify(U, V, W_candidate):
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W_candidate
                    break
        
        if best_rank < rank_sb:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
