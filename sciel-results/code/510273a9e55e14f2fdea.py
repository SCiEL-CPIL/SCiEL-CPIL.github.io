"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2 * 2x2 blocks, applied to 7 column pairs (14 columns)
    # plus schoolbook for the last column (1 column). Total rank = 7*7 + 4 = 53.
    n, m, p = DIMENSIONS
    # n=2, m=2, p=15
    # U shape: (n*m, rank) = (4, 53)
    # V shape: (m*p, rank) = (30, 53)
    # W shape: (p*n, rank) = (30, 53)
    
    rank_strassen = 7  # Strassen rank for each 2x2x2 block
    rank_schoolbook = 4  # Schoolbook rank for each 2x2x1 block (2 cols) -> actually 2*2*1=4 per column
    # Wait, schoolbook for 2x2 * 2x1 is 2*2*1 = 4? No.
    # A is 2x2, B is 2x15.
    # We split B into 7 blocks of 2x2 and 1 block of 2x1.
    # For each 2x2 block of B, we use Strassen (rank 7).
    # For the last 2x1 block, we use schoolbook.
    # Schoolbook for 2x2 * 2x1:
    #   C_0 = A_00 * B_0 + A_01 * B_1
    #   C_1 = A_10 * B_0 + A_11 * B_1
    # This requires 4 products: A_00*B_0, A_01*B_1, A_10*B_0, A_11*B_1.
    # So rank 4 for the last column.
    
    total_rank = 7 * rank_strassen + rank_schoolbook
    # total_rank = 49 + 4 = 53
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2 * 2x2:
    # U (A indices): 0=A00, 1=A01, 2=A10, 3=A11
    # V (B indices): 0=B00, 1=B01, 2=B10, 3=B11
    # W (C indices): 0=C00, 1=C01, 2=C10, 3=C11
    #
    # m1 = (A00 + A11) * (B00 + B11)
    # m2 = (A10 + A11) * B00
    # m3 = A00 * (B01 - B11)
    # m4 = A11 * (B10 - B00)
    # m5 = (A00 + A01) * B11
    # m6 = (A10 - A00) * (B00 + B01)
    # m7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = m1 - m4 - m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # U vectors (for A):
    # m1: [1, 0, 0, 1]
    # m2: [0, 0, 1, 1]
    # m3: [1, 0, 0, 0]
    # m4: [0, 0, 0, 1]
    # m5: [1, 1, 0, 0]
    # m6: [1, 0, -1, 0]
    # m7: [0, 1, 0, -1]
    
    # V vectors (for B):
    # m1: [1, 0, 0, 1]
    # m2: [1, 0, 0, 0]
    # m3: [0, 1, 0, -1]
    # m4: [1, 0, 0, -1]  # Wait, m4 = A11 * (B10 - B00) -> B indices: B10 - B00 = [-1, 0, 1, 0]
    # Let me re-check m4: A11 * (B10 - B00). B10 is index 2, B00 is index 0. So [-1, 0, 1, 0].
    # m5: [0, 0, 0, 1]
    # m6: [1, 1, 0, 0]  # (B00 + B01) -> [1, 1, 0, 0]
    # m7: [0, 0, 1, 1]  # (B10 + B11) -> [0, 0, 1, 1]
    
    # W vectors (for C):
    # m1: [1, 0, 0, 1]  # C00+=m1, C11+=m1
    # m2: [0, 0, 1, -1] # C10+=m2, C11-=m2
    # m3: [0, 1, 0, 1]  # C01+=m3, C11+=m3
    # m4: [-1, 0, 1, 0] # C00-=m4, C10+=m4
    # m5: [0, 1, 0, 0]  # C01+=m5
    # m6: [0, 0, 0, 1]  # C11+=m6
    # m7: [-1, 0, 0, 0] # C00-=m7
    
    U_strassen = [
        [1, 0, 1, 0, 1, 1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, -1, 0], # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    W_strassen = [
        [1, 0, 0, -1, 0, 0, -1], # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    # Fill Strassen blocks for 7 column pairs
    for block in range(7):
        base_col = block * 7
        for i in range(4):  # A indices
            for r in range(7):
                U[i][base_col + r] = U_strassen[i][r]
        for j in range(4):  # B indices (2 rows * 2 cols per block)
            for r in range(7):
                V[block * 4 + j][base_col + r] = V_strassen[j][r]
        for k in range(4):  # C indices (2 rows * 2 cols per block)
            for r in range(7):
                W[block * 4 + k][base_col + r] = W_strassen[k][r]
    
    # Fill schoolbook block for the last column (column index 14)
    base_col = 7 * 7  # 49
    # Schoolbook: 4 products
    # P1 = A00 * B0
    # P2 = A01 * B1
    # P3 = A10 * B0
    # P4 = A11 * B1
    # C0 = P1 + P2
    # C1 = P3 + P4
    
    # U: A00=1, A01=1, A10=1, A11=1
    U[0][base_col + 0] = 1  # A00 for P1
    U[1][base_col + 1] = 1  # A01 for P2
    U[2][base_col + 2] = 1  # A10 for P3
    U[3][base_col + 3] = 1  # A11 for P4
    
    # V: B0 for P1, B1 for P2, B0 for P3, B1 for P4
    # B column 14, rows 0 and 1
    V[14][base_col + 0] = 1  # B0 for P1
    V[15][base_col + 1] = 1  # B1 for P2
    V[14][base_col + 2] = 1  # B0 for P3
    V[15][base_col + 3] = 1  # B1 for P4
    
    # W: C0 = P1 + P2, C1 = P3 + P4
    # C indices: k*n + i, where k=14, i=0,1
    # C_14,0 = P1 + P2
    # C_14,1 = P3 + P4
    W[14 * 2 + 0][base_col + 0] = 1  # C0 += P1
    W[14 * 2 + 0][base_col + 1] = 1  # C0 += P2
    W[14 * 2 + 1][base_col + 2] = 1  # C1 += P3
    W[14 * 2 + 1][base_col + 3] = 1  # C1 += P4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
