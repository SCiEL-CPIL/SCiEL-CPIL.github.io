"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 9]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            # C[i,k] = sum_j A[i,j]*B[j,k]
            expected = 1  # The (i,j,k) entry of the multiplication tensor
            actual = 0
            for r in range(rank):
                actual += U[i*m + 0][r] * V[0*p + k][r] * W[k*n + i][r]
            # Actually, let's check the full tensor decomposition properly
    # Full verification: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if s != 1:
                    return False
    return True

def _try_rank(r, n, m, p, timeout=140.0):
    """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
    start_time = time.time()
    
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strategy: Fix U and V, solve for W, then check consistency.
    # Better: Use the fact that this is a bilinear problem.
    # 
    # Let's use a different approach: 
    # For rank r, we have r "gates". Each gate r contributes:
    #   sum over (i,j,k) of U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * delta_{T[i,j,k],1}
    # 
    # Actually, the condition is: for each (i,j,k):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This is a system of n*m*p equations in r*(n*m + m*p + p*n) unknowns.
    # But it's nonlinear (trilinear). 
    #
    # Alternative: Fix U and V, then W is determined by:
    #   W[k*n+i, r] = 1 / (U[i*m+j, r] * V[j*p+k, r]) for the (i,j,k) pair
    # But this must be consistent across all (i,j,k) pairs.
    #
    # Simpler approach: Enumerate small integer/half-integer values for U and V,
    # then solve for W and check consistency.
    
    # Given the small dimensions, let's try a more direct approach.
    # n=2, m=3, p=9: 54 tensor entries, each must equal 1.
    
    # For rank r, we have 3*r*n*m + r*m*p + r*p*n = r*(6*3 + 3*9 + 9*2) = r*(18+27+18) = r*63 variables
    # and 54 equations. For r >= 1, we have more variables than equations.
    
    # Let's try: fix U to have entries in {0, 1, -1, 1/2, -1/2} and similarly for V,
    # then solve for W.
    
    # Actually, let's use a smarter approach. The multiplication tensor for (2,3,9)
    # has structure. Let's try to find a decomposition using the fact that
    # we can combine some operations.
    
    # Try rank 30 first (a known bound might be around n*m*p - some savings)
    # Actually, for 2x3x9, the schoolbook is 54. Can we do better?
    
    # Let's try a randomized approach with local search.
    # Initialize with schoolbook, then try to merge pairs of gates.
    
    best_rank = n * m * p
    best_result = None
    
    # Start with schoolbook
    rank = best_rank
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Now try to reduce rank by merging gates
    # Two gates r1 and r2 can be merged if there exists a way to replace them
    # with a single gate that produces the same contribution to all tensor entries.
    
    # This is complex. Let's try a different strategy:
    # Use the Strassen-like idea: for 2x2x2, rank 7 < 8.
    # For our problem, let's try to find a lower rank by solving the system.
    
    # Let's try ranks from 40 down to 30
    for r in range(40, 29, -1):
        if time.time() - start_time > timeout - 10:
            break
        
        # Try to find a rank-r solution
        # Use a greedy approach: start with schoolbook (rank 54), try to eliminate gates
        
        # Actually, let's just return the best we can find quickly.
        # Given time constraints, let's try a few random initializations and gradient-free search.
        
        # For now, if we can't find better, return schoolbook
        pass
    
    # Return schoolbook as fallback
    U_list = [[U[i, j] for j in range(rank)] for i in range(n*m)]
    V_list = [[V[i, j] for j in range(rank)] for i in range(m*p)]
    W_list = [[W[i, j] for j in range(rank)] for i in range(p*n)]
    return U_list, V_list, W_list

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
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
# EVOLVE-BLOCK-END
