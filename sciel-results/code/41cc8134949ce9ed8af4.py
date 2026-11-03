"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 40  # 2 * 4 * 5
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Decompose A (4x5) into B0 (2x5) and B1 (2x5)
    # B0 = A[0:2, :5], B1 = A[2:4, :5]
    
    # Decompose B (5x5) into 2x2 blocks of size 2x2, with one 1x5 strip
    # Top 4 rows of B: 2 blocks of 2x2
    #   B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
    #   B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    # Bottom row of B: 1x5 strip B2 = B[4:5, 0:5]
    
    # C = A @ B (4x5)
    # C[0:2, 0:2] = B0 @ B00 + B1 @ B2[0:2]  (where B2[0:2] is B[4:5, 0:2])
    # C[0:2, 2:4] = B0 @ B01 + B1 @ B2[2:4]  (where B2[2:4] is B[4:5, 2:4])
    # C[0:2, 4:5] = B0 @ B2[4:5]             (where B2[4:5] is B[4:5, 4:5])
    # C[2:4, 0:2] = B0 @ B10 + B1 @ B2[0:2]
    # C[2:4, 2:4] = B0 @ B11 + B1 @ B2[2:4]
    # C[2:4, 4:5] = B0 @ B2[4:5]
    
    # For each 2x2 output block, use Strassen (rank 7)
    # For each 2x1 output column, use 2x2x1 decomposition (rank 4)
    
    # Let's index the rank-1 terms.
    # Strassen for 2x2x2 uses 7 terms.
    # For a 2x2 output block C_ij = A_block @ B_block:
    #   M1 = (A00 + A11) * (B00 + B11)
    #   M2 = (A10 + A11) * B00
    #   M3 = A00 * (B01 - B11)
    #   M4 = A11 * (B10 - B00)
    #   M5 = (A00 + A01) * B11
    #   M6 = (A10 - A00) * (B00 + B01)
    #   M7 = (A01 - A11) * (B10 + B11)
    
    # We'll build the decomposition block by block.
    # U has shape (4*5, 40) = (20, 40)
    # V has shape (5*5, 40) = (25, 40)
    # W has shape (5*4, 40) = (20, 40)
    
    # U row index: i * 5 + j, where i in 0..3, j in 0..4
    # V row index: j * 5 + k, where j in 0..4, k in 0..4
    # W row index: k * 4 + i, where k in 0..4, i in 0..3
    
    # Let's define helper functions to add Strassen terms for a 2x2 block.
    # A_block: rows i0, i0+1 of A, cols j0..j0+1
    # B_block: rows j0..j0+1 of B, cols k0..k0+1
    # C_block: rows i0, i0+1 of C, cols k0, k0+1
    
    def add_strassen(U, V, W, base_rank, a_rows, b_cols, c_rows, c_cols):
        # a_rows: list of 2 A-row indices (i0, i0+1)
        # b_cols: list of 2 B-col indices (k0, k0+1)
        # c_rows: list of 2 C-row indices (i0, i0+1)
        # c_cols: list of 2 C-col indices (k0, k0+1)
        # b_rows: list of 2 B-row indices (j0, j0+1)
        # Actually, we need to know the B rows too.
        pass
    
    # Let me restructure. For each 2x2 output block, I need to know:
    # - The 2 A-rows involved
    # - The 2 B-rows involved
    # - The 2 B-cols involved
    # - The 2 C-rows involved (same as A-rows)
    # - The 2 C-cols involved (same as B-cols)
    
    # For 2x2x2 Strassen with local indices:
    # A_local[2][2], B_local[2][2], C_local[2][2]
    # A_local[i][j] corresponds to A[a_rows[i]][b_rows[j]]
    # B_local[j][k] corresponds to B[b_rows[j]][b_cols[k]]
    # C_local[i][k] corresponds to C[c_rows[i]][c_cols[k]]
    
    def add_strassen_2x2x2(U, V, W, base, a_rows, b_rows, b_cols, c_rows, c_cols):
        # M1 = (A00 + A11) * (B00 + B11)
        #   U: A00, A11 -> 1
        #   V: B00, B11 -> 1
        #   W: C00, C11 -> 1
        r = base + 0
        U[a_rows[0] * m + b_rows[0]][r] += 1
        U[a_rows[1] * m + b_rows[1]][r] += 1
        V[b_rows[0] * p + b_cols[0]][r] += 1
        V[b_rows[1] * p + b_cols[1]][r] += 1
        W[b_cols[0] * n + c_rows[0]][r] += 1
        W[b_cols[1] * n + c_rows[1]][r] += 1
        
        # M2 = (A10 + A11) * B00
        r = base + 1
        U[a_rows[1] * m + b_rows[0]][r] += 1
        U[a_rows[1] * m + b_rows[1]][r] += 1
        V[b_rows[0] * p + b_cols[0]][r] += 1
        W[b_cols[0] * n + c_rows[1]][r] += 1
        
        # M3 = A00 * (B01 - B11)
        r = base + 2
        U[a_rows[0] * m + b_rows[0]][r] += 1
        V[b_rows[0] * p + b_cols[1]][r] += 1
        V[b_rows[1] * p + b_cols[1]][r] -= 1
        W[b_cols[1] * n + c_rows[0]][r] += 1
        
        # M4 = A11 * (B10 - B00)
        r = base + 3
        U[a_rows[1] * m + b_rows[1]][r] += 1
        V[b_rows[1] * p + b_cols[0]][r] += 1
        V[b_rows[0] * p + b_cols[0]][r] -= 1
        W[b_cols[0] * n + c_rows[1]][r] += 1
        
        # M5 = (A00 + A01) * B11
        r = base + 4
        U[a_rows[0] * m + b_rows[0]][r] += 1
        U[a_rows[0] * m + b_rows[1]][r] += 1
        V[b_rows[1] * p + b_cols[1]][r] += 1
        W[b_cols[1] * n + c_rows[0]][r] += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        r = base + 5
        U[a_rows[1] * m + b_rows[0]][r] += 1
        U[a_rows[0] * m + b_rows[0]][r] -= 1
        V[b_rows[0] * p + b_cols[0]][r] += 1
        V[b_rows[0] * p + b_cols[1]][r] += 1
        W[b_cols[0] * n + c_rows[1]][r] += 1
        W[b_cols[1] * n + c_rows[1]][r] += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        r = base + 6
        U[a_rows[0] * m + b_rows[1]][r] += 1
        U[a_rows[1] * m + b_rows[1]][r] -= 1
        V[b_rows[1] * p + b_cols[0]][r] += 1
        V[b_rows[1] * p + b_cols[1]][r] += 1
        W[b_cols[0] * n + c_rows[0]][r] += 1
        W[b_cols[1] * n + c_rows[0]][r] += 1
    
    # For 2x2x1 (2 A-rows, 2 B-rows, 1 B-col), rank 4:
    # C[i][k] = sum_j A[i][j] * B[j][k]
    # This is just 2 terms for each i, so 4 terms total.
    # Actually, it's just schoolbook: 2*2*1 = 4 terms.
    
    def add_2x2x1(U, V, W, base, a_rows, b_rows, b_col, c_rows, c_col):
        for t in range(4):
            r = base + t
            i = t // 2
            j = t % 2
            U[a_rows[i] * m + b_rows[j]][r] += 1
            V[b_rows[j] * p + b_col][r] += 1
            W[b_col * n + c_rows[i]][r] += 1
    
    # Block (0,0): C[0:2, 0:2] = B0 @ B00 + B1 @ B2[0:2]
    # B0 = A[0:2, :], B00 = B[0:2, 0:2]
    # B1 = A[2:4, :], B2[0:2] = B[4:5, 0:2]
    # This is a sum of two 2x2x2 products. We can combine them.
    # C[0:2, 0:2] = A[0:2, 0:2] @ B[0:2, 0:2] + A[0:2, 2:4] @ B[2:4, 0:2] + A[0:2, 4:5] @ B[4:5, 0:2]
    # Wait, A is 4x5, so B0 = A[0:2, 0:5], which has columns 0,1,2,3,4.
    # B00 = B[0:2, 0:2]
    # B10 = B[2:4, 0:2]
    # B20 = B[4:5, 0:2]
    # C[0:2, 0:2] = A[0:2, 0:2] @ B00 + A[0:2, 2:4] @ B10 + A[0:2, 4:5] @ B20
    # This is a 2x5x2 @ 5x2x2 product.
    # We can decompose this using 2x2x2 Strassen for the first two 2x2 blocks,
    # and 2x2x1 for the last 1x2 block.
    
    # Let's just use the schoolbook for the whole 4x5x5 but with some Strassen optimization.
    # Actually, let's use a simpler approach: split A into two 2x5 halves.
    # For each half, compute its product with B using a 2x5x5 decomposition.
    # 2x5x5 can be done with rank 2*5*5 = 50, but we can do better.
    # For 2x2x2 sub-blocks, use Strassen (7 instead of 8).
    
    # Let's do this:
    # C[0:2, 0:2] = A[0:2,0:2]@B[0:2,0:2] + A[0:2,2:4]@B[2:4,0:2] + A[0:2,4:5]@B[4:5,0:2]
    #   = Strassen(A[0:2,0:2], B[0:2,0:2]) + Strassen(A[0:2,2:4], B[2:4,0:2]) + 2x2x1(A[0:2,4:5], B[4:5,0:2])
    #   = 7 + 7 + 4 = 18 terms
    # Similarly for other blocks.
    
    # C[0:2, 0:2]:
    add_strassen_2x2x2(U, V, W, 0, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    add_strassen_2x2x2(U, V, W, 7, [0, 1], [2, 3], [0, 1], [0, 1], [0, 1])
    add_2x2x1(U, V, W, 14, [0, 1], [4], [0], [0, 1], [0])
    add_2x2x1(U, V, W, 18, [0, 1], [4], [1], [0, 1], [1])
    
    # C[0:2, 2:4]:
    add_strassen_2x2x2(U, V, W, 20, [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    add_strassen_2x2x2(U, V, W, 27, [0, 1], [2, 3], [2, 3], [0, 1], [2, 3])
    add_2x2x1(U, V, W, 34, [0, 1], [4], [2], [0, 1], [2])
    add_2x2x1(U, V, W, 38, [0, 1], [4], [3], [0, 1], [3])
    
    # C[0:2, 4:5]:
    add_2x2x1(U, V, W, 40, [0, 1], [0], [4], [0, 1], [4])
    add_2x2x1(U, V, W, 44, [0, 1], [1], [4], [0, 1], [4])
    add_2x2x1(U, V, W, 48, [0, 1], [2], [4], [0, 1], [4])
    add_2x2x1(U, V, W, 52, [0, 1], [3], [4], [0, 1], [4])
    add_2x2x1(U, V, W, 56, [0, 1], [4], [4], [0, 1], [4])
    
    # Wait, this is getting too many terms. Let me reconsider.
    
    # Actually, the rank should be 40. Let me think of a better decomposition.
    # 4x5x5: we can view it as 2x(2x5x5).
    # 2x5x5 has rank at most 50, but we can do better.
    # For 2x2x2, Strassen gives rank 7.
    # For 2x5x5, we can split the 5 into 2+2+1.
    # 2x2x2 (rank 7) + 2x2x2 (rank 7) + 2x2x1 (rank 4) + 2x1x2 (rank 4) + 2x1x1 (rank 2)
    # Wait, that's for the B matrix columns.
    
    # Let's just use the schoolbook for now, which gives rank 100.
    # The task says the current program has score 1.0, which means rank 100.
    # We need to beat that.
    
    # Let me restart with a clean approach.
    rank = 100
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
