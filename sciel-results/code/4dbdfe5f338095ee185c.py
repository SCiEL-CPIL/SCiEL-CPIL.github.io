import time
import random

def _strassen_2x2x2_factors():
    """
    Returns U, V, W factors for 2x2x2 matrix multiplication using Strassen's algorithm.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    U[i*2+j, r]
    V[j*2+k, r]
    W[k*2+i, r]
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Strassen multipliers
    # M1 = A00 * (B01 - B11)
    U[0][0] = 1  # A00
    V[1][0] = 1  # B01
    V[3][0] = -1 # B11
    W[1][0] = 1  # C01
    W[3][0] = -1 # C11
    
    # M2 = (A00 + A10) * B11
    U[0][1] = 1  # A00
    U[2][1] = 1  # A10
    V[3][1] = 1  # B11
    W[0][1] = -1 # C00
    W[2][1] = 1  # C10
    
    # M3 = (A10 - A00) * (B00 + B10)
    U[2][2] = 1  # A10
    U[0][2] = -1 # A00
    V[0][2] = 1  # B00
    V[2][2] = 1  # B10
    W[0][2] = 1  # C00
    W[2][2] = 1  # C10
    
    # M4 = A11 * (B10 - B00)
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1 # B00
    W[0][3] = 1  # C00
    W[2][3] = -1 # C10
    
    # M5 = (A00 + A11) * (B00 + B11)
    U[0][4] = 1  # A00
    U[3][4] = 1  # A11
    V[0][4] = 1  # B00
    V[3][4] = 1  # B11
    W[0][4] = 1  # C00
    W[1][4] = 1  # C01
    W[2][4] = 1  # C10
    W[3][4] = -1 # C11
    
    # M6 = (A01 - A11) * (B10 + B11)
    U[1][5] = 1  # A01
    U[3][5] = -1 # A11
    V[2][5] = 1  # B10
    V[3][5] = 1  # B11
    W[0][5] = 1  # C00
    W[3][5] = 1  # C11
    
    # M7 = (A00 + A01) * B01
    U[0][6] = 1  # A00
    U[1][6] = 1  # A01
    V[1][6] = 1  # B01
    W[1][6] = 1  # C01
    W[3][6] = -1 # C11
    
    return U, V, W

def _outer_product_2x1x1x7_factors():
    """
    Returns factors for 2x1 * 1x7 -> 2x7.
    Rank 14.
    U: (2, 14), V: (7, 14), W: (14, 14) -> W is (p*n, rank) = (14, 14)
    Wait, W index is k*n+i. Here n=2, p=7? No.
    A is 2x1. B is 1x7. C is 2x7.
    n=2, m=1, p=7.
    U: (2*1, 14) = (2, 14)
    V: (1*7, 14) = (7, 14)
    W: (7*2, 14) = (14, 14)
    """
    rank = 14
    U = [[0] * rank for _ in range(2)]
    V = [[0] * rank for _ in range(7)]
    W = [[0] * rank for _ in range(14)]
    
    col = 0
    for i in range(2):
        for k in range(7):
            U[i][col] = 1
            V[k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
    return U, V, W

def _matvec_1x2x2x7_factors():
    """
    Returns factors for 1x2 * 2x7 -> 1x7.
    Rank 14.
    A is 1x2. B is 2x7. C is 1x7.
    n=1, m=2, p=7.
    U: (1*2, 14) = (2, 14)
    V: (2*7, 14) = (14, 14)
    W: (7*1, 14) = (7, 14)
    """
    rank = 14
    U = [[0] * rank for _ in range(2)]
    V = [[0] * rank for _ in range(14)]
    W = [[0] * rank for _ in range(7)]
    
    col = 0
    for j in range(2):
        for k in range(7):
            U[j][col] = 1
            V[j*7 + k][col] = 1
            W[k*1 + 0][col] = 1
            col += 1
    return U, V, W

def _vecmul_1x1x1x7_factors():
    """
    Returns factors for 1x1 * 1x7 -> 1x7.
    Rank 7.
    A is 1x1. B is 1x7. C is 1x7.
    n=1, m=1, p=7.
    U: (1, 7)
    V: (7, 7)
    W: (7, 7)
    """
    rank = 7
    U = [[0] * rank for _ in range(1)]
    V = [[0] * rank for _ in range(7)]
    W = [[0] * rank for _ in range(7)]
    
    for k in range(7):
        U[0][k] = 1
        V[k][k] = 1
        W[k][k] = 1
    return U, V, W

def construct():
    """
    Constructs a rank-60 decomposition for 3x3 * 3x7 matrix multiplication.
    """
    n, m, p = 3, 3, 7
    
    # Total rank 60
    # 3 Strassen blocks (2x2x2) for the 2x2 part of A and first 6 cols of B?
    # No, B is 3x7.
    # Split A into 2x2, 2x1, 1x2, 1x1.
    # Split B into 2x7, 1x7.
    # C_TL (2x7) = A_TL(2x2) * B_TL(2x7) + A_TR(2x1) * B_BL(1x7)
    # C_BL (1x7) = A_BL(1x2) * B_TL(2x7) + A_BR(1x1) * B_BL(1x7)
    
    # A_TL * B_TL: 2x2 * 2x7.
    # Split B_TL into three 2x2 blocks and one 2x1 block.
    # B_TL = [B_0 (2x2), B_1 (2x2), B_2 (2x2), B_3 (2x1)]
    # Rank = 3 * R(2,2,2) + R(2,2,1) = 3*7 + 4 = 25.
    
    # A_TR * B_BL: 2x1 * 1x7. Rank 14.
    # A_BL * B_TL: 1x2 * 2x7. Rank 14.
    # A_BR * B_BL: 1x1 * 1x7. Rank 7.
    
    # Total Rank = 25 + 14 + 14 + 7 = 60.
    
    total_rank = 60
    U = [[0] * total_rank for _ in range(9)] # 3*3
    V = [[0] * total_rank for _ in range(21)] # 3*7
    W = [[0] * total_rank for _ in range(21)] # 7*3
    
    col_offset = 0
    
    # 1. A_TL * B_TL (2x2 * 2x7)
    # Split B_TL cols 0-5 into 3 blocks of 2, and col 6 is separate?
    # B_TL is rows 0,1 of B. Cols 0-6.
    # We need 2x2 * 2x2 for cols 0-1, 2-3, 4-5.
    # And 2x2 * 2x1 for col 6.
    
    # Block 1: Cols 0-1
    U1, V1, W1 = _strassen_2x2x2_factors()
    for i in range(4):
        for r in range(7):
            U[i][col_offset + r] += U1[i][r]
            # V index: j*2+k. j is 0,1. k is 0,1.
            V[i//2 * 2 + i%2 * 0 + 0][col_offset + r] # Wrong indexing
            # V row is j*p + k? No, in local 2x2x2, p=2.
            # Global V row is j*7 + k.
            # Local j, k map to global j, k.
            # U1 row i maps to global i.
            # V1 row j*2+k maps to global j*7 + k.
            # W1 row k*2+i maps to global k*3 + i.
            
    # Let's do this carefully with a helper function to add factors
    
    def add_factors(U_local, V_local, W_local, row_offset_U, row_offset_V, row_offset_W, col_offset):
        """
        Adds local factors to global U, V, W.
        U_local: (n_l*m_l, rank)
        V_local: (m_l*p_l, rank)
        W_local: (p_l*n_l, rank)
        """
        n_l = int((row_offset_U - (row_offset_U % 1)) / 1) # Not easy to infer n_l, m_l, p_l from offsets alone
        # We pass dimensions explicitly in the calling logic instead.
        pass

    # I will implement the addition manually in the construct function for clarity and speed.
    
    # Reset U, V, W
    U = [[0] * total_rank for _ in range(9)]
    V = [[0] * total_rank for _ in range(21)]
    W = [[0] * total_rank for _ in range(21)]
    
    col = 0
    
    # Part 1: A_TL (2x2) * B_TL (2x7)
    # B_TL cols 0-1: 2x2x2 Strassen
    U1, V1, W1 = _strassen_2x2x2_factors()
    for i in range(4):
        for r in range(7):
            U[i][col+r] += U1[i][r]
            j = i // 2
            k = i % 2
            # V1 row is j*2 + k_local. k_local is 0 or 1.
            # Global V row is j*7 + k_global.
            # Here k_global is 0 or 1.
            V[j*7 + k][col+r] += V1[j*2 + k][r]
            # W1 row is k*2 + i_local.
            # Global W row is k_global*3 + i_global.
            W[k*3 + i][col+r] += W1[k*2 + i][r]
    col += 7
    
    # B_TL cols 2-3: 2x2x2 Strassen
    U1, V1, W1 = _strassen_2x2x2_factors()
    for i in range(4):
        for r in range(7):
            U[i][col+r] += U1[i][r]
            j = i // 2
            k = i % 2
            V[j*7 + (k+2)][col+r] += V1[j*2 + k][r]
            W[(k+2)*3 + i][col+r] += W1[k*2 + i][r]
    col += 7
    
    # B_TL cols 4-5: 2x2x2 Strassen
    U1, V1, W1 = _strassen_2x2x2_factors()
    for i in range(4):
        for r in range(7):
            U[i][col+r] += U1[i][r]
            j = i // 2
            k = i % 2
            V[j*7 + (k+4)][col+r] += V1[j*2 + k][r]
            W[(k+4)*3 + i][col+r] += W1[k*2 + i][r]
    col += 7
    
    # B_TL col 6: 2x2x1 (Rank 4)
    # A_TL (2x2) * B_col6 (2x1)
    # C_col6 (2x1)
    # Rank 4.
    # U: (4, 4), V: (2, 4), W: (2, 4)
    # U[i*2+j, r], V[j*1+k, r] -> V[j, r], W[k*2+i, r] -> W[k*2+i, r]
    # Terms:
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # r=0: A00 B0 -> C0
    # r=1: A01 B1 -> C0
    # r=2: A10 B0 -> C1
    # r=3: A11 B1 -> C1
    U[0][col+0] = 1
    V[0][col+0] = 1
    W[0*2+0][col+0] = 1
    
    U[1][col+1] = 1
    V[1][col+1] = 1
    W[0*2+0][col+1] = 1
    
    U[2][col+2] = 1
    V[0][col+2] = 1
    W[1*2+0][col+2] = 1
    
    U[3][col+3] = 1
    V[1][col+3] = 1
    W[1*2+0][col+3] = 1
    col += 4
    
    # Part 2: A_TR (2x1) * B_BL (1x7)
    # A_TR rows 0,1 col 2.
    # B_BL row 2 cols 0-6.
    # C_TL rows 0,1 cols 0-6.
    # Rank 14.
    U2, V2, W2 = _outer_product_2x1x1x7_factors()
    # U2 row i (0,1) maps to global U row i*3 + 2.
    # V2 row k (0-6) maps to global V row 2*7 + k.
    # W2 row k*2+i maps to global W row k*3 + i.
    for i in range(2):
        for r in range(14):
            U[i*3 + 2][col+r] += U2[i][r]
            # V2 row k. k = r // 2? No, V2 is 7x14.
            # In _outer_product, V[k][col] = 1.
            # So V2 row k corresponds to global V row 14 + k.
            # W2 row k*2+i corresponds to global W row k*3 + i.
            pass # The helper function returns full matrices, I need to map them.
            
    # Let's just use the helper's output directly by mapping indices.
    # U2 is 2x14. Global U rows 2, 5.
    # V2 is 7x14. Global V rows 14-20.
    # W2 is 14x14. Global W rows 0-20? No, W is 21x14.
    # W2 row k*2+i. Global W row k*3+i.
    
    for i in range(2):
        for r in range(14):
            U[i*3 + 2][col+r] += U2[i][r]
    for k in range(7):
        for r in range(14):
            V[14 + k][col+r] += V2[k][r]
    for k in range(7):
        for i in range(2):
            for r in range(14):
                W[k*3 + i][col+r] += W2[k*2 + i][r]
    col += 14
    
    # Part 3: A_BL (1x2) * B_TL (2x7)
    # A_BL row 2 cols 0,1.
    # B_TL rows 0,1 cols 0-6.
    # C_BL row 2 cols 0-6.
    # Rank 14.
    U3, V3, W3 = _matvec_1x2x2x7_factors()
    # U3 is 2x14. Global U rows 6, 7.
    # V3 is 14x14. Global V rows 0-13.
    # W3 is 7x14. Global W rows 6-12? No, W row k*3+2.
    
    for j in range(2):
        for r in range(14):
            U[6 + j][col+r] += U3[j][r]
    for j in range(2):
        for k in range(7):
            for r in range(14):
                V[j*7 + k][col+r] += V3[j*7 + k][r]
    for k in range(7):
        for r in range(14):
            W[k*3 + 2][col+r] += W3[k][r]
    col += 14
    
    # Part 4: A_BR (1x1) * B_BL (1x7)
    # A_BR row 2 col 2.
    # B_BL row 2 cols 0-6.
    # C_BL row 2 cols 0-6.
    # Rank 7.
    U4, V4, W4 = _vecmul_1x1x1x7_factors()
    # U4 is 1x7. Global U row 8.
    # V4 is 7x7. Global V rows 14-20.
    # W4 is 7x7. Global W rows 6-12.
    
    for r in range(7):
        U[8][col+r] += U4[0][r]
    for k in range(7):
        for r in range(7):
            V[14 + k][col+r] += V4[k][r]
    for k in range(7):
        for r in range(7):
            W[k*3 + 2][col+r] += W4[k][r]
    col += 7
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Quick validation
    res = construct()
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = 3, 3, 7
    rank = len(U[0])
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Rank: {rank}, Valid: {valid}")
