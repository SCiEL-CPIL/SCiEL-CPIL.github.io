"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]
START_TIME = time.time()
TIME_LIMIT = 120.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x3x7 matrix multiplication.
    Returns schoolbook decomposition (rank 42) as the safe fallback.
    """
    n, m, p = DIMENSIONS
    
    # Attempt to find a lower-rank decomposition using structured search
    result = _try_lower_rank(n, m, p)
    if result is not None:
        return result
    
    # Fallback: schoolbook decomposition (rank 42)
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

def _try_lower_rank(n, m, p):
    """
    Attempts to find a lower-rank decomposition using structured patterns.
    Returns None if no valid lower-rank decomposition is found within time budget.
    """
    if _time_remaining() < 10:
        return None
    
    # Try various ranks below 42
    for rank in range(33, 42):
        if _time_remaining() < 5:
            break
        result = _search_structured(n, m, p, rank)
        if result is not None:
            return result
    
    return None

def _search_structured(n, m, p, rank):
    """
    Searches for a rank-r decomposition using structured sparse patterns.
    For small ranks, we can try to find patterns that satisfy the tensor equations.
    """
    # For 2x3x7, the tensor has 42 non-zero entries (i,j,k) where 0<=i<2, 0<=j<3, 0<=k<7
    # We need: sum_t U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t] = 1 for all (i,j,k)
    
    # Strategy: Try to find a decomposition where each rank-1 term covers multiple (i,j,k) triples
    # This is equivalent to finding a covering of the 42 triples by rank-1 tensors
    
    # For now, this is a placeholder that returns None
    # A full implementation would use combinatorial search or integer programming
    return None

# EVOLVE-BLOCK-END
