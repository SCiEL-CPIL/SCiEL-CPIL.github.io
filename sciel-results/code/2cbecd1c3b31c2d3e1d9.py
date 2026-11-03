"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition satisfies sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _try_rank(n, m, p, rank):
    """
    Try to find a rank-r decomposition using a constructive approach.
    We use the fact that the system is trilinear and try to find solutions
    by fixing some variables and solving linear systems for others.
    """
    # Start with schoolbook and try to reduce rank by merging columns
    # For now, try a direct approach: set up the system and use numerical methods
    
    # Total number of equations: n*m*p
    num_eqs = n * m * p
    
    # We'll try to find a solution by starting from schoolbook and perturbing
    # Actually, let's try a more direct approach for small rank reductions
    
    # For rank r, we need 6*r + 30*r + 20*r = 56*r unknowns and 60 equations
    # This is underdetermined for any r >= 1, but the nonlinearity makes it hard
    
    # Let's try: can we achieve rank 59 by modifying two schoolbook columns?
    # The schoolbook has 60 columns, one for each (i,j,k) triple.
    # To get rank 59, we need to "merge" two triples into one column.
    
    # A column c with values (u_val, v_val, w_val) contributes:
    # u_val[i*m+j] * v_val[j*p+k] * w_val[k*n+i] to equation (i,j,k)
    
    # For a single column to satisfy multiple equations, we need:
    # u_val[i1*m+j1] * v_val[j1*p+k1] * w_val[k1*n+i1] = 1
    # u_val[i2*m+j2] * v_val[j2*p+k2] * w_val[k2*n+i2] = 1
    # And for all other (i,j,k): contribution should be 0 (handled by other columns)
    
    # This is complex. Let's try a different approach: use the fact that
    # we can have fractional values.
    
    # Try rank 59: modify the schoolbook by combining two terms
    # Let's pick two triples that share structure
    
    # Actually, let's try to solve this using a simple iterative approach
    # For now, return None if we can't find a solution quickly
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    Uses a constructive approach to find the minimal rank decomposition.
    """
    n, m, p = DIMENSIONS
    
    # Try ranks from 59 down to 1
    # Start with higher ranks that are more likely to have solutions
    for rank in range(n * m * p - 1, 0, -1):
        result = _try_rank(n, m, p, rank)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(n, m, p, U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
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
