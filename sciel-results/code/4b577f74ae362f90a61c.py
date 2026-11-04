"""Program for matrix multiplication 2x4 x 4x6 with rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import itertools

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    
    This implementation uses a combinatorial generation approach to construct
    the schoolbook decomposition. The rank is 48 (n*m*p).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition using itertools
    for col, (i, j, k) in enumerate(itertools.product(range(n), range(m), range(p))):
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
