"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition as (U, V, W) lists."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2(n, m, p):
    """
    Try to reduce rank by applying a 2x2 Strassen split to the p dimension
    (treating p as 2 * (p//2) if p is even and p >= 2).
    This is a heuristic: we embed the 3x3x(p//2) problem twice and add
    the 6 extra Strassen terms.
    """
    if p < 2 or p % 2 != 0:
        return None
    p2 = p // 2
    if p2 < 1:
        return None
    # We need a decomposition for 3x3xp2. Use schoolbook as base for now.
    # A full Strassen embedding would combine two copies of 3x3xp2 with 6 extra terms.
    # Rank = 2 * (rank of 3x3xp2) + 6 * (n*m) ... this is complex.
    # For simplicity, we just return None to avoid incorrect embeddings.
    # A proper Strassen embedding for 3x3xp where p is even:
    # Split C (n x p) into C1 (n x p2) and C2 (n x p2).
    # Split B (m x p) into B1 (m x p2) and B2 (m x p2).
    # C1 = A*B1, C2 = A*B2.
    # This doesn't directly allow Strassen unless we also split A or B.
    # Strassen works best when both input dimensions are splittable.
    # Since n=3, m=3, p=8, we can split p=8 into 4+4, but we can't split n or m into 2+1 easily for standard Strassen.
    # So we skip this.
    return None


def _try_strassen_block(n, m, p):
    """
    Attempt to find a lower-rank decomposition by exploiting block structure.
    For 3x3x8, we can view the 8 columns as 4 pairs.
    This is a placeholder for a more sophisticated search.
    """
    # For now, just return the schoolbook decomposition.
    return _schoolbook(n, m, p)


def construct():
    """
    Construct the factor matrices for the matrix multiplication tensor.
    Returns a dict with 'U', 'V', 'W' keys.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with the schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    best_rank = len(U[0])
    
    # Try to improve with Strassen-like embeddings
    # For 3x3x8, we can try to split p=8 into 2+2+2+2 and use Strassen on pairs
    # This is a complex task, so we'll just return the schoolbook for now.
    # In a real implementation, we would search for better decompositions.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
