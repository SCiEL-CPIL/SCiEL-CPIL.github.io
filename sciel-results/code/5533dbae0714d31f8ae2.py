"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose T_{3,3,8} into two T_{3,3,4} blocks.
    # T_{3,3,4} is T_{3,3,3} + T_{3,3,1}.
    # Rank(T_{3,3,3}) = 27 (Schoolbook)
    # Rank(T_{3,3,1}) = 1
    # Rank(T_{3,3,4}) = 28
    # Total Rank = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a schoolbook 3x3x3 block
    # offset_k: starting k index for this block (0 or 4)
    # offset_col: starting column index in U,V,W
    def add_block(offset_k, offset_col):
        # 3x3x3 schoolbook
        for i in range(n):
            for j in range(m):
                for k_local in range(3):
                    col = offset_col + i * m * 3 + j * 3 + k_local
                    k_global = offset_k + k_local
                    U[i * m + j][col] = 1
                    V[j * p + k_global][col] = 1
                    W[k_global * n + i][col] = 1
        # 3x3x1 rank 1 block
        # k_local = 3, so k_global = offset_k + 3
        col = offset_col + 27
        k_global = offset_k + 3
        # U[i*m+j] = 1 for all i,j
        # V[j*p+k] = 1 for all j
        # W[k*n+i] = 1 for all i
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
        for j in range(m):
            V[j * p + k_global][col] = 1
        for i in range(n):
            W[k_global * n + i][col] = 1

    # Add first block (k=0..3)
    add_block(0, 0)
    # Add second block (k=4..7)
    add_block(4, 28)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
