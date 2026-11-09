"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-60 (schoolbook) decomposition for 3x4x5 matrix multiplication.
    Uses index mapping from rank index r to (i, j, k) indices.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        # Decompose r into i, j, k
        # i is the major index for A and C (row of A, row of C)
        # j is the inner index (col of A, row of B)
        # k is the minor index for B and C (col of B, col of C)
        # Mapping: r = i * (m * p) + j * p + k
        i = r // (m * p)
        remainder = r % (m * p)
        j = remainder // p
        k = remainder % p
        
        # Calculate linear indices for U, V, W
        # U index: i * m + j
        u_idx = i * m + j
        # V index: j * p + k
        v_idx = j * p + k
        # W index: k * n + i  (NOTE: Convention W[k*n+i, r])
        w_idx = k * n + i
        
        U[u_idx][r] = 1
        V[v_idx][r] = 1
        W[w_idx][r] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
