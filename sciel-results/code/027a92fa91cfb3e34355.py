"""Kronecker-style decomposition of 3x5x6 matrix multiplication."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 6
    # Decompose m=5 as 2+3, p=6 as 2+4
    # Block 1: 3x2x2 (rank 7)
    # Block 2: 3x3x2 (rank 12)
    # Block 3: 3x2x4 (rank 12)
    # Block 4: 3x3x4 (rank 19)
    
    rank = 7 + 12 + 12 + 19
    U = [[0] * rank for _ in range(n * m)]  # 15 x rank
    V = [[0] * rank for _ in range(m * p)]  # 30 x rank
    W = [[0] * rank for _ in range(p * n)]  # 18 x rank
    
    def add_block(U, V, W, offset, u_rows, u_cols, u_block, v_rows, v_cols, v_block, w_rows, w_cols, w_block):
        """Add a sub-block to the global matrices at the given offset."""
        for i in range(len(u_rows)):
            for c in range(len(u_cols)):
                U[u_rows[i]][offset + c] += u_block[i][c]
        for i in range(len(v_rows)):
            for c in range(len(v_cols)):
                V[v_rows[i]][offset + c] += v_block[i][c]
        for i in range(len(w_rows)):
            for c in range(len(w_cols)):
                W[w_rows[i]][offset + c] += w_block[i][c]
    
    # Block 1: 3x2x2, rows i=0..2, j=0..1, k=0..1
    # Strassen for 2x2x2, padded to 3x2x2
    # U: 3x2 = 6 rows, 7 cols
    # V: 2x2 = 4 rows, 7 cols
    # W: 2x3 = 6 rows, 7 cols
    u_rows_1 = [i*2 + j for i in range(3) for j in range(2)]  # 0..5
    u_block_1 = [
        [1, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 1, 0, 0, 0, 0],  # (1,0)
        [0, 0, 0, 1, 0, 0, 0],  # (1,1)
        [0, 0, 0, 0, 1, 0, 0],  # (2,0)
        [0, 0, 0, 0, 0, 1, 0],  # (2,1)
    ]
    # For 2x2x2 Strassen, the standard decomposition:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # For 3x2x2, we just use the first 2x2 part with Strassen
    v_rows_1 = [j*2 + k for j in range(2) for k in range(2)]  # 0..3
    v_block_1 = [
        [1, 1, 0, 0, 0, 0, 0],  # (0,0)
        [1, 1, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 1, 0, 0, 0, 1],  # (1,0)
        [0, 0, 1, 0, 0, 0, 1],  # (1,1)
    ]
    w_rows_1 = [k*3 + i for k in range(2) for i in range(3)]  # 0..5
    w_block_1 = [
        [1, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 0, 0, 0, 1, 0, 0],  # (0,1)
        [0, 1, 0, 0, 0, 0, 0],  # (0,2)
        [1, 0, 0, 0, 0, 0, 0],  # (1,0)
        [0, 0, 0, 0, 1, 0, 0],  # (1,1)
        [0, 1, 0, 0, 0, 0, 0],  # (1,2)
    ]
    # This isn't right. Let me use schoolbook for simplicity first, then optimize.
    
    # Actually, let me just use schoolbook for all blocks to ensure correctness first.
    # Rank for schoolbook: 3*2*2 + 3*3*2 + 3*2*4 + 3*3*4 = 12 + 18 + 24 + 36 = 90
    
    # Use block decomposition with optimized ranks
    # Split m=5 into 2+3, p=6 into 2+4
    # Block (0,0): 3x2x2, rank 7 (Strassen for 2x2x2, padded)
    # Block (0,1): 3x2x4, rank 12
    # Block (1,0): 3x3x2, rank 12  
    # Block (1,1): 3x3x4, rank 19
    
    total_rank = 7 + 12 + 12 + 19
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def add_block(offset, u_rows, u_block, v_rows, v_block, w_rows, w_block):
        for i in range(len(u_rows)):
            for c in range(len(u_block[0])):
                U[u_rows[i]][offset + c] += u_block[i][c]
        for i in range(len(v_rows)):
            for c in range(len(v_block[0])):
                V[v_rows[i]][offset + c] += v_block[i][c]
        for i in range(len(w_rows)):
            for c in range(len(w_block[0])):
                W[w_rows[i]][offset + c] += w_block[i][c]
    
    # Block (0,0): i=0..2, j=0..1, k=0..1 (3x2x2)
    # Use Strassen for the 2x2 part, with i as a free parameter
    # Rank 7: 3*2 + 1 extra for Strassen
    # Actually, for 3x2x2, we can use rank 6 (3 independent 1x2x2, each rank 2)
    # Or use Strassen: treat as 3 copies of 1x2x2
    # 1x2x2 has rank 2 (since it's just 2 scalars times a 2-vector)
    # So 3x2x2 has rank 3*2 = 6
    u_rows_00 = [i * 2 + j for i in range(3) for j in range(2)]  # 0..5
    u_block_00 = [[0]*6 for _ in range(6)]
    for i in range(3):
        for j in range(2):
            u_block_00[i*2+j][i*2+j] = 1  # This is wrong, let me redo
    
    # Simpler: use schoolbook for each block
    # Block (0,0): 3x2x2, rank 12
    # Block (0,1): 3x2x4, rank 24
    # Block (1,0): 3x3x2, rank 18
    # Block (1,1): 3x3x4, rank 36
    # Total: 90
    
    # Let me just use schoolbook with rank 90 for correctness
    rank = n * m * p  # 90
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
