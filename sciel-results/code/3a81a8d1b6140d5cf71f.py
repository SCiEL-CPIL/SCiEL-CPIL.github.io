"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Tensor decomposition for 2x2 by 2x12 matrix multiplication.
    Uses a 'row-aggregation' approach.
    
    The multiplication C = A * B where A is 2x2 and B is 2x12.
    C is 2x12.
    C[i, k] = sum_j A[i, j] * B[j, k].
    
    We decompose this by iterating over the output rows i (0, 1).
    For a fixed i, C[i, :] is a 1x12 vector computed from A[i, :] (1x2) and B (2x12).
    This sub-problem is a 1x2 by 2x12 matrix multiplication.
    The rank of 1x2 by 2x12 is 2*12 = 24.
    Since the two output rows i=0 and i=1 are independent (disjoint output indices),
    the total rank is 24 + 24 = 48.
    
    Indices:
    U: (n*m, rank) = (4, 48). Index u = i*m + j.
    V: (m*p, rank) = (24, 48). Index v = j*p + k.
    W: (p*n, rank) = (24, 48). Index w = k*n + i.
    """
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
