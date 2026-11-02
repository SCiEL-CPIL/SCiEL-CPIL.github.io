"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Schoolbook Algorithm for guaranteed correctness.
    # Rank is n * m * p
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over all possible terms in the tensor decomposition
    # Term k corresponds to the product A[i,j] * B[j,l] contributing to C[i,l]
    for i in range(n):
        for j in range(m):
            for l in range(p):
                # Calculate the rank index (column in U, V, W)
                k = i * m * p + j * p + l
                
                # U: Input A has shape (n, m). Index is (i, j).
                # Row in U is i * m + j
                row_u = i * m + j
                U[row_u][k] = 1
                
                # V: Input B has shape (m, p). Index is (j, l).
                # Row in V is j * p + l
                row_v = j * p + l
                V[row_v][k] = 1
                
                # W: Output C has shape (n, p). Index is (i, l).
                # Convention: W[k*n+i, r] implies row index is l * n + i 
                # (based on failed code logic: row_w = (b_col_start + l_sub) * 3 + i_sub)
                # Here l is the column index of C, i is the row index of C.
                row_w = l * n + i
                W[row_w][k] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
