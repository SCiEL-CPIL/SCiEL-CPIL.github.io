def construct():
    n, m, p = 3, 4, 6
    
    # New partition strategy:
    # Split A rows into 2+1
    # For top 2 rows (2x4) * B (4x6):
    #   Split A cols 1+3, B rows 1+3
    #   (2x1)*(1x6) rank 2, (2x3)*(3x6) rank 9 (3 blocks of 2x3*3x2 rank 3)
    #   Subtotal: 11
    # For bottom row (1x4) * B (4x6):
    #   6 blocks of (1x2)*(2x2) rank 4
    #   Subtotal: 24
    # Total: 35
    
    # Schoolbook rank is n*m*p = 3*4*6 = 72
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_2x1_1x6(a_r, b_r, c_r):
        """Add (2x1)*(1x6) block (schoolbook rank 12)."""
        nonlocal col
        for i in range(2):
            for j in range(6):
                U[(a_r + i) * m + 0][col] += 1
                V[b_r * p + j][col] += 1
                W[j * n + (c_r + i)][col] += 1
                col += 1

    def add_2x3_3x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add (2x3)*(3x2) block (schoolbook rank 12)."""
        nonlocal col
        for i in range(2):
            for k in range(3):
                for j in range(2):
                    U[(a_r + i) * m + (a_c + k)][col] += 1
                    V[(b_r + k) * p + (b_c + j)][col] += 1
                    W[(c_c + j) * n + (c_r + i)][col] += 1
                    col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 1x2*2x2 block (schoolbook rank 4)."""
        nonlocal col
        for k in range(2):
            for j in range(2):
                U[a_r * m + (a_c + k)][col] += 1
                V[(b_r + k) * p + (b_c + j)][col] += 1
                W[(c_c + j) * n + c_r][col] += 1
                col += 1

    # Top 2 rows: A[0:2, :] * B
    # Split A cols 1+3, B rows 1+3
    # Part 1: A[0:2, 0] (2x1) * B[0, :] (1x6) -> C[0:2, :]
    add_2x1_1x6(0, 0, 0)
    
    # Part 2: A[0:2, 1:4] (2x3) * B[1:4, :] (3x6) -> C[0:2, :]
    # Split B cols 2+2+2
    for j in range(3):
        add_2x3_3x2(0, 1, 1, j * 2, 0, j * 2)
    
    # Bottom row: A[2, :] * B
    # 6 blocks of (1x2)*(2x2)
    for j in range(3):
        add_1x2_2x2(2, 0, 0, j * 2, 2, j * 2)   # A[2,0:2] * B[0:2, j*2:j*2+2]
        add_1x2_2x2(2, 2, 2, j * 2, 2, j * 2)   # A[2,2:4] * B[2:4, j*2:j*2+2]

    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
