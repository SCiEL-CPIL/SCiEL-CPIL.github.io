"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 66 decomposition
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for 2x2x2
    # a_idx: list of 4 U indices (A00, A01, A10, A11)
    # b_idx: list of 4 V indices (B00, B01, B10, B11)
    # c_idx: list of 4 W indices (C00, C01, C10, C11)
    def add_strassen(a_idx, b_idx, c_idx, start_col):
        # M1 = (A00+A11)(B00+B11)
        U[a_idx[0]][start_col] += 1
        U[a_idx[3]][start_col] += 1
        V[b_idx[0]][start_col] += 1
        V[b_idx[3]][start_col] += 1
        W[c_idx[0]][start_col] += 1
        W[c_idx[3]][start_col] += 1
        W[c_idx[1]][start_col] += 1
        W[c_idx[2]][start_col] -= 1
        col_local = start_col + 1
        
        # M2 = (A10+A11)B00
        U[a_idx[2]][col_local] += 1
        U[a_idx[3]][col_local] += 1
        V[b_idx[0]][col_local] += 1
        W[c_idx[1]][col_local] -= 1
        W[c_idx[2]][col_local] += 1
        col_local += 1
        
        # M3 = A00(B01-B11)
        U[a_idx[0]][col_local] += 1
        V[b_idx[1]][col_local] += 1
        V[b_idx[3]][col_local] -= 1
        W[c_idx[0]][col_local] -= 1
        W[c_idx[1]][col_local] += 1
        col_local += 1
        
        # M4 = A11(B10-B00)
        U[a_idx[3]][col_local] += 1
        V[b_idx[2]][col_local] += 1
        V[b_idx[0]][col_local] -= 1
        W[c_idx[0]][col_local] += 1
        W[c_idx[2]][col_local] += 1
        col_local += 1
        
        # M5 = (A00+A01)B11
        U[a_idx[0]][col_local] += 1
        U[a_idx[1]][col_local] += 1
        V[b_idx[3]][col_local] += 1
        W[c_idx[0]][col_local] -= 1
        W[c_idx[1]][col_local] += 1
        col_local += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a_idx[2]][col_local] += 1
        U[a_idx[0]][col_local] -= 1
        V[b_idx[0]][col_local] += 1
        V[b_idx[1]][col_local] += 1
        W[c_idx[3]][col_local] += 1
        col_local += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a_idx[1]][col_local] += 1
        U[a_idx[3]][col_local] -= 1
        V[b_idx[2]][col_local] += 1
        V[b_idx[3]][col_local] += 1
        W[c_idx[0]][col_local] += 1
        W[c_idx[3]][col_local] += 1
        col_local += 1
        
        return col_local

    # Helper to add a Schoolbook block for 1x2x2
    # a_idx: list of 2 U indices (A0, A1)
    # b_idx: list of 4 V indices (B00, B01, B10, B11)
    # c_idx: list of 2 W indices (C0, C1)
    def add_schoolbook_1x2x2(a_idx, b_idx, c_idx, start_col):
        # C0 = A0*B00 + A1*B10
        # C1 = A0*B01 + A1*B11
        
        # Col 1: A0*B00 -> C0
        U[a_idx[0]][start_col] = 1
        V[b_idx[0]][start_col] = 1
        W[c_idx[0]][start_col] = 1
        col_local = start_col + 1
        
        # Col 2: A1*B10 -> C0
        U[a_idx[1]][col_local] = 1
        V[b_idx[2]][col_local] = 1
        W[c_idx[0]][col_local] = 1
        col_local += 1
        
        # Col 3: A0*B01 -> C1
        U[a_idx[0]][col_local] = 1
        V[b_idx[1]][col_local] = 1
        W[c_idx[1]][col_local] = 1
        col_local += 1
        
        # Col 4: A1*B11 -> C1
        U[a_idx[1]][col_local] = 1
        V[b_idx[3]][col_local] = 1
        W[c_idx[1]][col_local] = 1
        col_local += 1
        
        return col_local

    # Loop over the 3 k-blocks (width 2)
    for q in range(3):
        k_start = q * 2
        
        # Define indices for this k-block
        # A indices (3x4):
        # Top 2 rows: i=0,1
        # Bot 1 row: i=2
        # Left 2 cols: j=0,1
        # Right 2 cols: j=2,3
        
        # A11 (top-left 2x2): i in {0,1}, j in {0,1}
        a11 = [i * 4 + j for i in [0, 1] for j in [0, 1]]
        # A12 (top-right 2x2): i in {0,1}, j in {2,3}
        a12 = [i * 4 + j for i in [0, 1] for j in [2, 3]]
        # A21 (bot-left 1x2): i=2, j in {0,1}
        a21 = [2 * 4 + 0, 2 * 4 + 1]
        # A22 (bot-right 1x2): i=2, j in {2,3}
        a22 = [2 * 4 + 2, 2 * 4 + 3]
        
        # B indices (4x6):
        # Left 2 rows: j=0,1
        # Right 2 rows: j=2,3
        # Cols: k in {k_start, k_start+1}
        
        # B1q (top-left 2x2 for this k): j in {0,1}, k in {k_start, k_start+1}
        b1q = [j * 6 + k for j in [0, 1] for k in [k_start, k_start + 1]]
        # B2q (top-right 2x2 for this k): j in {2,3}, k in {k_start, k_start+1}
        b2q = [j * 6 + k for j in [2, 3] for k in [k_start, k_start + 1]]
        
        # C indices (3x6):
        # Top 2 rows: i=0,1
        # Bot 1 row: i=2
        # Cols: k in {k_start, k_start+1}
        
        # C1q (top 2x2): i in {0,1}, k in {k_start, k_start+1}
        c1q = [k * 3 + i for k in [k_start, k_start + 1] for i in [0, 1]]
        # C2q (bot 1x2): i=2, k in {k_start, k_start+1}
        c2q = [k * 3 + 2 for k in [k_start, k_start + 1]]
        
        # 1. Strassen for A11 * B1q -> C1q (top part)
        col = add_strassen(a11, b1q, c1q, col)
        
        # 2. Strassen for A12 * B2q -> C1q (top part)
        col = add_strassen(a12, b2q, c1q, col)
        
        # 3. Schoolbook for A21 * B1q -> C2q (bot part)
        col = add_schoolbook_1x2x2(a21, b1q, c2q, col)
        
        # 4. Schoolbook for A22 * B2q -> C2q (bot part)
        col = add_schoolbook_1x2x2(a22, b2q, c2q, col)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
