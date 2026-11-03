"""Program for finding rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose the 2x3x8 matrix multiplication tensor using a block decomposition.
    
    Strategy:
    - Split the 3x8 matrix B into two 3x4 blocks: B1 (cols 0-3), B2 (cols 4-7)
    - C1 = A * B1 (2x4), C2 = A * B2 (2x4)
    - Decompose each 2x3x4 multiplication with rank 22.
    
    Total rank: 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen 2x2x2 block
    # A_sub: 2x2 matrix of indices in U
    # B_sub: 2x2 matrix of indices in V
    # C_sub: 2x2 matrix of indices in W
    # offset: column index in U, V, W
    def add_strassen(a_idx, b_idx, c_idx, offset):
        # a_idx[i][j] corresponds to U[i*m + j]
        # b_idx[i][j] corresponds to V[i*p + j]
        # c_idx[i][j] corresponds to W[j*n + i]
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[a_idx[0][0]][offset] += 1
        U[a_idx[1][1]][offset] += 1
        V[b_idx[0][0]][offset] += 1
        V[b_idx[1][1]][offset] += 1
        W[c_idx[0][0]][offset] += 1
        W[c_idx[1][1]][offset] -= 1
        W[c_idx[1][0]][offset] += 1
        W[c_idx[0][1]][offset] += 1
        # Note: The W coefficients for M1 are 1, -1, 1, 1 for Z11, Z12, Z21, Z22?
        # Z11 = M1 + M4 - M5 + M7
        # Z12 = M3 + M5
        # Z21 = M2 + M4
        # Z22 = M1 - M2 + M3 + M6
        # So for M1:
        # W[0,0] (Z11) += 1
        # W[0,1] (Z12) += 0
        # W[1,0] (Z21) += 0
        # W[1,1] (Z22) += 1
        # Wait, the indices in W are [k*n + i].
        # c_idx[0][0] is W[0*n + 0] = W[0] -> Z11
        # c_idx[0][1] is W[1*n + 0] = W[2] -> Z12
        # c_idx[1][0] is W[0*n + 1] = W[1] -> Z21
        # c_idx[1][1] is W[1*n + 1] = W[3] -> Z22
        
        # Let's re-map carefully.
        # W index = k*n + i.
        # c_idx[i][j] where i is row of C (0,1), j is col of C (0,1).
        # W index for C[i][j] is j*n + i.
        
        # M1 contributes to Z11 (1) and Z22 (1)
        W[c_idx[0][0]][offset] += 1 # Z11
        W[c_idx[1][1]][offset] += 1 # Z22
        
        # M2 = (a10 + a11) * b00
        U[a_idx[1][0]][offset + 1] += 1
        U[a_idx[1][1]][offset + 1] += 1
        V[b_idx[0][0]][offset + 1] += 1
        # M2 contributes to Z21 (1) and Z22 (-1)
        W[c_idx[1][0]][offset + 1] += 1 # Z21
        W[c_idx[1][1]][offset + 1] -= 1 # Z22
        
        # M3 = a00 * (b01 - b11)
        U[a_idx[0][0]][offset + 2] += 1
        V[b_idx[0][1]][offset + 2] += 1
        V[b_idx[1][1]][offset + 2] -= 1
        # M3 contributes to Z12 (1) and Z22 (1)
        W[c_idx[0][1]][offset + 2] += 1 # Z12
        W[c_idx[1][1]][offset + 2] += 1 # Z22
        
        # M4 = a11 * (b10 - b00)
        U[a_idx[1][1]][offset + 3] += 1
        V[b_idx[1][0]][offset + 3] += 1
        V[b_idx[0][0]][offset + 3] -= 1
        # M4 contributes to Z11 (1) and Z21 (1)
        W[c_idx[0][0]][offset + 3] += 1 # Z11
        W[c_idx[1][0]][offset + 3] += 1 # Z21
        
        # M5 = (a00 + a01) * b11
        U[a_idx[0][0]][offset + 4] += 1
        U[a_idx[0][1]][offset + 4] += 1
        V[b_idx[1][1]][offset + 4] += 1
        # M5 contributes to Z11 (-1) and Z12 (1)
        W[c_idx[0][0]][offset + 4] -= 1 # Z11
        W[c_idx[0][1]][offset + 4] += 1 # Z12
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[a_idx[1][0]][offset + 5] += 1
        U[a_idx[0][0]][offset + 5] -= 1
        V[b_idx[0][0]][offset + 5] += 1
        V[b_idx[0][1]][offset + 5] += 1
        # M6 contributes to Z22 (1)
        W[c_idx[1][1]][offset + 5] += 1 # Z22
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[a_idx[0][1]][offset + 6] += 1
        U[a_idx[1][1]][offset + 6] -= 1
        V[b_idx[1][0]][offset + 6] += 1
        V[b_idx[1][1]][offset + 6] += 1
        # M7 contributes to Z11 (1)
        W[c_idx[0][0]][offset + 6] += 1 # Z11
        
        return offset + 7

    # Process Block 1 (cols 0-3 of B)
    # A_L: rows 0-1, cols 0-1 of A
    a_idx_1 = [[0, 1], [3, 4]] # U indices: i*m+j. 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
    # B_TL: rows 0-1, cols 0-1 of B1 (which are cols 0-1 of B)
    b_idx_tl = [[0, 1], [8, 9]] # V indices: i*p+j. 0*8+0=0, 0*8+1=1, 1*8+0=8, 1*8+1=9
    # C_TL: rows 0-1, cols 0-1 of C1 (which are cols 0-1 of C)
    c_idx_tl = [[0, 1], [2, 3]] # W indices: j*n+i. 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3
    
    offset = add_strassen(a_idx_1, b_idx_tl, c_idx_tl, 0)
    
    # B_TR: rows 0-1, cols 2-3 of B1 (which are cols 2-3 of B)
    b_idx_tr = [[2, 3], [10, 11]] # V indices: 0*8+2=2, 0*8+3=3, 1*8+2=10, 1*8+3=11
    # C_TR: rows 0-1, cols 2-3 of C1
    c_idx_tr = [[4, 5], [6, 7]] # W indices: 2*2+0=4, 2*2+1=5, 3*2+0=6, 3*2+1=7
    
    offset = add_strassen(a_idx_1, b_idx_tr, c_idx_tr, offset)
    
    # A_R: rows 0-1, col 2 of A
    # B_B: row 2, cols 0-3 of B1
    # 8 multiplications
    for k in range(4):
        # C[0][k] += A[0][2] * B[2][k]
        U[2][offset] = 1 # A[0][2] index 0*3+2=2
        V[2*8 + k][offset] = 1 # B[2][k] index 2*8+k
        W[k*2 + 0][offset] = 1 # C[0][k] index k*2+0
        offset += 1
        
        # C[1][k] += A[1][2] * B[2][k]
        U[5][offset] = 1 # A[1][2] index 1*3+2=5
        V[2*8 + k][offset] = 1
        W[k*2 + 1][offset] = 1 # C[1][k] index k*2+1
        offset += 1
        
    # Process Block 2 (cols 4-7 of B)
    # A_L is the same
    a_idx_2 = [[0, 1], [3, 4]]
    
    # B_TL2: rows 0-1, cols 4-5 of B
    b_idx_tl2 = [[4, 5], [12, 13]] # V indices: 0*8+4=4, 0*8+5=5, 1*8+4=12, 1*8+5=13
    # C_TL2: rows 0-1, cols 4-5 of C
    c_idx_tl2 = [[8, 9], [10, 11]] # W indices: 4*2+0=8, 4*2+1=9, 5*2+0=10, 5*2+1=11
    
    offset = add_strassen(a_idx_2, b_idx_tl2, c_idx_tl2, offset)
    
    # B_TR2: rows 0-1, cols 6-7 of B
    b_idx_tr2 = [[6, 7], [14, 15]] # V indices: 0*8+6=6, 0*8+7=7, 1*8+6=14, 1*8+7=15
    # C_TR2: rows 0-1, cols 6-7 of C
    c_idx_tr2 = [[12, 13], [14, 15]] # W indices: 6*2+0=12, 6*2+1=13, 7*2+0=14, 7*2+1=15
    
    offset = add_strassen(a_idx_2, b_idx_tr2, c_idx_tr2, offset)
    
    # A_R is the same
    for k in range(4, 8):
        # C[0][k] += A[0][2] * B[2][k]
        U[2][offset] = 1
        V[2*8 + k][offset] = 1
        W[k*2 + 0][offset] = 1
        offset += 1
        
        # C[1][k] += A[1][2] * B[2][k]
        U[5][offset] = 1
        V[2*8 + k][offset] = 1
        W[k*2 + 1][offset] = 1
        offset += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
