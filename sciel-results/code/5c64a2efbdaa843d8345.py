"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a schoolbook tensor decomposition for matrix multiplication
    of a 3x3 matrix by a 3x8 matrix.
    
    The tensor T_{i,j,k} = 1 for all i in [0,3), j in [0,3), k in [0,8)
    represents the multiplication: (AB)_{i,k} = sum_j A_{i,j} * B_{j,k}
    
    Schoolbook decomposition:
    U[i*m + j, c] = 1 if c == i*m + j
    V[j*p + k, c] = 1 if c == i*m + j (where c encodes i,j)
    W[k*n + i, c] = 1 if c == i*m + j (where c encodes i,j)
    
    Index conventions:
    - U has shape (n*m, rank) = (9, rank), indexed by (i,j)
    - V has shape (m*p, rank) = (24, rank), indexed by (j,k)
    - W has shape (p*n, rank) = (24, rank), indexed by (k,i)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 3*3*8 = 72
    
    # Build factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 9 x 72
    V = [[0] * rank for _ in range(m * p)]  # 24 x 72
    W = [[0] * rank for _ in range(p * n)]  # 24 x 72
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
