"""
Program for 6x6x6 Matrix Multiplication Tensor Decomposition.
Uses a nested block decomposition:
1. Decompose 6x6 matrices into 3x3 blocks of 2x2 submatrices.
2. Use Schoolbook multiplication for the 3x3 block structure (Rank 27).
3. Use Schoolbook multiplication for the 2x2 submatrix structure (Rank 4).
Total Rank = 27 * 4 = 108.
"""

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the factor matrices U, V, W for the rank-108 decomposition 
    of the 6x6 matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS
    # Block dimensions
    # Outer blocks: 3x3
    # Inner blocks: 2x2
    outer_dim = 3
    inner_dim = 2
    
    # Rank is product of ranks of outer and inner schoolbook multiplications
    rank_outer = outer_dim * outer_dim * outer_dim # 27
    rank_inner = inner_dim * inner_dim * inner_dim # 4
    total_rank = rank_outer * rank_inner # 108
    
    # Initialize matrices with zeros
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # Iterate over outer schoolbook terms (I, K, J)
    for I in range(outer_dim):
        for K in range(outer_dim):
            for J in range(outer_dim):
                # Iterate over inner schoolbook terms (i, k, j)
                for i in range(inner_dim):
                    for k in range(inner_dim):
                        for j in range(inner_dim):
                            # Calculate global indices
                            # U index: row = I*2 + i, col = K*2 + k
                            u_row = (I * inner_dim + i) * m + (K * inner_dim + k)
                            
                            # V index: row = K*2 + k, col = J*2 + j
                            v_row = (K * inner_dim + k) * p + (J * inner_dim + j)
                            
                            # W index: col = J*2 + j, row = I*2 + i
                            # Convention: W[k*n + i] where k is col index, i is row index
                            w_row = (J * inner_dim + j) * n + (I * inner_dim + i)
                            
                            U[u_row][col] = 1
                            V[v_row][col] = 1
                            W[w_row][col] = 1
                            
                            col += 1
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
