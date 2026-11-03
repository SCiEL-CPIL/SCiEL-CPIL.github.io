"""Program for 3x3x8 matrix multiplication tensor decomposition."""

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x3 matrix by a 3x8 matrix.
    
    The trivial decomposition has rank 72 = 3*3*8.
    We attempt to find a lower-rank decomposition using structured patterns.
    
    For matrix multiplication (n x m) * (m x p) -> (n x p),
    the tensor T has entries T[i,j,k] = 1 if the output (i,k) depends on input (j).
    
    We use the standard identity: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    The trivial rank is n*m*p. We look for improvements using:
    1. Kronecker product structures
    2. Known low-rank decompositions for small cases
    
    For 3x3x8, we try a decomposition that exploits the structure.
    
    One approach: use the fact that 3x3 multiplication can be done with rank 23
    (Lader's algorithm) or similar, but for 3x3x8 we need to extend.
    
    Actually, let's use a systematic approach: we'll construct a decomposition
    that may have rank less than 72 by exploiting linear dependencies.
    
    For now, we return the trivial rank-72 decomposition but with a more
    compact construction that could be improved.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Try to find a lower rank decomposition
    # The trivial rank is n*m*p = 72
    
    # We'll use a structured approach: for each output position (i,k),
    # we need to compute sum_j A[i,j] * B[j,k]
    
    # One idea: group by i and k, and use shared intermediate computations.
    
    # For a 3x3x8 problem, let's try rank = 45 or similar
    # by exploiting that some products can be shared.
    
    # Actually, let's just return the trivial decomposition for correctness,
    # but structured in a way that's easy to verify.
    
    rank = n * m * p  # 72
    
    # Build U, V, W as lists of lists
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
