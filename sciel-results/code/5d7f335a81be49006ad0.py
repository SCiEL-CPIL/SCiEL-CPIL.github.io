"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-44 exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    
    Strategy: 
    1. Partition B (4x6) into B1 (4x3) and B2 (4x3).
    2. For each 4x3 block, partition into Left (4x2) and Right (4x1).
    3. For Left (4x2): Split A (2x4) into AL (2x2) and AR (2x2).
       Split B_left (4x2) into T1 (2x2) and T2 (2x2).
       Compute AL*T1 (Rank 7) and AR*T2 (Rank 7). Total 14.
    4. For Right (4x1): Split A (2x4) into AL (2x2) and AR (2x2).
       Split B_right (4x1) into T1 (2x1) and T2 (2x1).
       Compute AL*T1 (Rank 4) and AR*T2 (Rank 4). Total 8.
    
    Total Rank = 2 * (14 + 8) = 44.
    """
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen 2x2x2
    # a_col_start: starting column in A (0 or 2)
    # b_row_start: starting row in B (0 or 2)
    # b_col_start: starting column in B (0, 1, 3, 4)
    # c_col_start: starting column in C (0, 1, 3, 4)
    # offset: current column index in U, V, W
    def add_strassen_222(offset, a_col_start, b_row_start, b_col_start, c_col_start):
        # Strassen coefficients for U (A)
        # m1: A00+A11, m2: A10+A11, m3: A00, m4: A11, m5: A00+A01, m6: A10-A00, m7: A01-A11
        u_coeffs = [
            [1, 1, 1, 0, 1, -1, 0], # i=0, j=0 (A00)
            [0, 1, 0, 1, 0, 1, 1],  # i=1, j=0 (A10)
            [1, 0, 1, 0, 1, 0, 1],  # i=0, j=1 (A01)
            [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1 (A11)
        ]
        
        # Strassen coefficients for V (B)
        # m1: B00+B11, m2: B00, m3: B01-B11, m4: B10-B00, m5: B11, m6: B00+B01, m7: B10+B11
        v_coeffs = [
            [1, 1, 1, -1, 0, 1, 0], # j=0, k=0 (B00)
            [1, 0, -1, 1, 1, 0, 1], # j=1, k=0 (B10)
            [0, 1, 1, 0, 0, 1, 0],  # j=0, k=1 (B01)
            [1, 0, -1, 0, 1, 0, 1]  # j=1, k=1 (B11)
        ]
        
        # Strassen coefficients for W (C)
        # m1: C00+C11, m2: -C10+C11, m3: C01+C11, m4: C00+C10, m5: -C00+C01, m6: C11, m7: C00
        w_coeffs = [
            [1, -1, 1, 1, -1, 0, 1], # i=0, k=0 (C00)
            [0, 1, 0, 0, 1, 0, 0],   # i=1, k=0 (C10)
            [1, 0, 1, 0, 1, 0, 0],   # i=0, k=1 (C01)
            [1, 1, 1, 0, 0, 1, 0]    # i=1, k=1 (C11)
        ]
        
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    val = u_coeffs[i*2+j][r]
                    if val != 0:
                        u_idx = i * 4 + (a_col_start + j)
                        U[u_idx][offset + r] += val
            
            for j in range(2):
                for k in range(2):
                    val = v_coeffs[j*2+k][r]
                    if val != 0:
                        v_idx = (b_row_start + j) * 6 + (b_col_start + k)
                        V[v_idx][offset + r] += val
            
            for i in range(2):
                for k in range(2):
                    val = w_coeffs[i*2+k][r]
                    if val != 0:
                        w_idx = (c_col_start + k) * 2 + i
                        W[w_idx][offset + r] += val

    # Helper to add Trivial 2x2x1
    # a_col_start: 0 or 2
    # b_row_start: 0 or 2
    # b_col_start: column in B
    # c_col_start: column in C
    def add_trivial_221(offset, a_col_start, b_row_start, b_col_start, c_col_start):
        # m1: A00*B0 -> C0
        # m2: A01*B1 -> C0
        # m3: A10*B0 -> C1
        # m4: A11*B1 -> C1
        
        u_coeffs = [
            [1, 0, 0, 0], # A00
            [0, 1, 0, 0], # A01
            [0, 0, 1, 0], # A10
            [0, 0, 0, 1]  # A11
        ]
        
        v_coeffs = [
            [1, 0, 1, 0], # B0
            [0, 1, 0, 1]  # B1
        ]
        
        w_coeffs = [
            [1, 1, 0, 0], # C0
            [0, 0, 1, 1]  # C1
        ]
        
        for r in range(4):
            for i in range(2):
                for j in range(2):
                    val = u_coeffs[i*2+j][r]
                    if val != 0:
                        u_idx = i * 4 + (a_col_start + j)
                        U[u_idx][offset + r] += val
            
            for j in range(2):
                val = v_coeffs[j][r]
                if val != 0:
                    v_idx = (b_row_start + j) * 6 + b_col_start
                    V[v_idx][offset + r] += val
            
            for i in range(2):
                val = w_coeffs[i][r]
                if val != 0:
                    w_idx = c_col_start * 2 + i
                    W[w_idx][offset + r] += val

    # Block 1: B cols 0-2
    # Left: Cols 0-1
    add_strassen_222(0, 0, 0, 0, 0)   # AL * B11
    add_strassen_222(7, 2, 2, 0, 0)   # AR * B21
    # Right: Col 2
    add_trivial_221(14, 0, 0, 2, 2)   # AL * B12
    add_trivial_221(18, 2, 2, 2, 2)   # AR * B22

    # Block 2: B cols 3-5
    # Left: Cols 3-4
    add_strassen_222(22, 0, 0, 3, 3)  # AL * B11
    add_strassen_222(29, 2, 2, 3, 3)  # AR * B21
    # Right: Col 5
    add_trivial_221(36, 0, 0, 5, 5)   # AL * B12
    add_trivial_221(40, 2, 2, 5, 5)   # AR * B22

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
