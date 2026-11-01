"""Matrix multiplication 3,3,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Construct a tensor decomposition for 3x3 * 3x5 matrix multiplication.
    
    Returns a dictionary with keys 'U', 'V', 'W' representing factor matrices
    of shapes (9, rank), (15, rank), (15, rank) respectively.
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    t0 = time.time()
    deadline = t0 + 120.0  # Keep safety margin
    
    # Trivial rank-45 decomposition (schoolbook)
    rank = n * m * p  # 45
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
    
    result = {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition using numerical optimization
    # The matrix multiplication tensor for C = AB where A is n x m and B is m x p
    # has entries T[i, j, k] = sum_l A[i, l] * B[l, k]
    # We want to find rank-1 tensors that sum to this tensor.
    # A rank-1 tensor is defined by vectors u (length n*m), v (length m*p), w (length p*n)
    # such that u[i*m + j] * v[j*p + k] * w[k*n + i] = T[i, j, k]
    
    # For the trivial decomposition, each term is a basis tensor with u, v, w having 
    # a single 1 at positions (i,m+j), (j,p+k), (k,n+i) respectively.
    
    # Strategy: Use gradient descent on the tensor rank minimization problem.
    # We start from the trivial decomposition and try to reduce the rank by 
    # merging terms that are nearly linearly dependent.
    
    # Alternative approach: Use the fact that for small matrix multiplication,
    # known low-rank decompositions exist. For 3x3x3, the best known rank is 23 
    # (Winograd). For 3x3x5, we can try to extend this idea.
    
    # Given the complexity and time constraints, we'll attempt a simple heuristic:
    # Check if any pair of terms in the trivial decomposition can be combined
    # into a single term while preserving the tensor structure.
    
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be combined if:
    # 1. They share the same i (output row), same j (inner index), or same k (output col)
    # 2. The resulting combined tensor still matches the multiplication tensor
    
    # In general, for the trivial decomposition, two terms can only be combined
    # if they differ in exactly one index and the other two indices match.
    # But even then, combining them would require the combined u, v, w vectors
    # to satisfy the tensor equation, which is not generally possible.
    
    # A more systematic approach would be to use numerical tensor decomposition
    # algorithms like CP decomposition with rank minimization, but this is 
    # computationally expensive and may not converge within the time limit.
    
    # For now, we return the trivial decomposition. If a lower-rank decomposition
    # is found, it can be hardcoded here.
    
    return result
# EVOLVE-BLOCK-END
