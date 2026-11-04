"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _naive_decomposition(n, m, p):
    """Return the standard (rank = n*m*p) exact decomposition."""
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
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check that the decomposition is exact: for all i,j,k,
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True


def _try_block_decomposition(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition using block decomposition.
    
    For 4x5x6, we can try to split the matrices into blocks and use
    Strassen's algorithm for 2x2x2 blocks where applicable.
    
    Strategy:
    - Split A (4x5) into A1 (2x5) and A2 (2x5)
    - Split B (5x6) into B1 (5x2), B2 (5x2), B3 (5x2)
    - C = A1*B1 + A1*B2 + A1*B3 + A2*B1 + A2*B2 + A2*B3
    - Each product is 2x2, so we can use Strassen (rank 7) for each
    - Total rank: 6 * 7 = 42
    
    But wait, the dimensions don't match for Strassen. Strassen requires
    square matrices. Let's think differently.
    
    Actually, for 4x5x6:
    - We can pad to 6x6x6 or use a different approach.
    - Or we can use the fact that rank(4x5x6) <= rank(6x6x6) = 81 (best known)
      but that's not directly helpful.
    
    Let's try a simpler block approach:
    - Split A into 2x5 blocks: A1 (2x5), A2 (2x5)
    - Split B into 5x2 blocks: B1 (5x2), B2 (5x2), B3 (5x2)
    - Each A_i * B_j is 2x2, naive rank 4, Strassen rank 7
    - But Strassen is worse for 2x2! Naive is 4, Strassen is 7.
    
    So for 2x2x2, naive (4) is better than Strassen (7).
    
    For larger blocks:
    - 3x3x3: naive 27, best known 23
    - 4x4x4: naive 64, best known 49
    
    For 4x5x6, let's try:
    - Split A (4x5) into A1 (3x5) and A2 (1x5)
    - Split B (5x6) into B1 (5x4) and B2 (5x2)
    - A1*B1: 3x5*5x4, naive 60
    - A1*B2: 3x5*5x2, naive 30
    - A2*B1: 1x5*5x4, naive 20
    - A2*B2: 1x5*5x2, naive 10
    - Total: 120, same as naive.
    
    It seems like block decomposition with naive sub-products doesn't help.
    
    Let's try using known lower-rank decompositions for specific block sizes.
    For 3x3x3, rank 23 is known. Can we use that?
    
    Actually, the problem is that the blocks need to be square for known
    decompositions. For 4x5x6, we don't have a clean square block structure.
    
    Given the difficulty, let's just return None and use the naive decomposition.
    """
    return None


def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS
    
    # First, get the naive decomposition
    U, V, W = _naive_decomposition(n, m, p)
    rank = n * m * p
    
    # Verify the naive decomposition
    if not _verify(U, V, W, n, m, p):
        # This should never happen, but just in case
        pass
    
    # Try to find a lower-rank decomposition
    result = _try_block_decomposition(n, m, p, deadline)
    if result is not None:
        U, V, W = result
        rank = len(U[0])
    
    # Convert to lists for the return format (already lists)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
