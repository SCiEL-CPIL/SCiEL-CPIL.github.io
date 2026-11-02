"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for A_L (2x2) * B_LL (2x2)
    # A_L indices in U: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B_LL indices in V: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4 (Wait, B is 3x3, so indices are row*3+col)
    # B is 3x3. B_LL is rows 0,1 and cols 0,1.
    # B indices: 0,1,2 (row 0), 3,4,5 (row 1), 6,7,8 (row 2)
    # B_LL: 0,1,3,4
    
    # M1 = (a00+a11)(b00+b11)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[4][col] = 1
    # W output for C_LL[0,0] and C_LL[1,1]
    # C = A B. C[i,k].
    # C_LL is C[0:2, 0:2].
    # W index is k*n + i.
    # C[0,0] -> W[0*2+0] = W[0]
    # C[1,1] -> W[1*2+1] = W[3]
    W[0][col] = 1
    W[3][col] = 1
    col += 1
    
    # M2 = (a10+a11)b00
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    # C[1,0] -> W[0*2+1] = W[1]
    W[1][col] = 1
    col += 1
    
    # M3 = a00(b01-b11)
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    # C[0,1] -> W[1*2+0] = W[2]
    W[2][col] = 1
    col += 1
    
    # M4 = a11(b10-b00)
    U[3][col] = 1
    V[3][col] = 1; V[0][col] = -1
    # C[0,0] -> W[0], C[1,0] -> W[1]
    W[0][col] = 1
    W[1][col] = 1
    col += 1
    
    # M5 = (a00+a01)b11
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    # C[0,0] -> W[0] (sub), C[0,1] -> W[2] (add)
    W[0][col] = -1
    W[2][col] = 1
    col += 1
    
    # M6 = (a10-a00)(b00+b01)
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    # C[1,1] -> W[3] (add)
    W[3][col] = 1
    col += 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1][col] = 1; U[3][col] = -1
    V[3][col] = 1; V[4][col] = 1
    # C[0,0] -> W[0] (add), C[1,1] -> W[3] (sub)
    W[0][col] = 1
    W[3][col] = -1
    col += 1
    
    # A_R (2x1) * B_RL (1x2)
    # A_R: a02 (idx 4), a12 (idx 5)
    # B_RL: b20 (idx 6), b21 (idx 7)
    # P1 = a02 b20 -> C[0,0]
    U[4][col] = 1
    V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # P2 = a02 b21 -> C[0,1]
    U[4][col] = 1
    V[7][col] = 1
    W[2][col] = 1
    col += 1
    
    # P3 = a12 b20 -> C[1,0]
    U[5][col] = 1
    V[6][col] = 1
    W[1][col] = 1
    col += 1
    
    # P4 = a12 b21 -> C[1,1]
    U[5][col] = 1
    V[7][col] = 1
    W[3][col] = 1
    col += 1
    
    # A_L (2x2) * B_LR (2x1)
    # B_LR: b02 (idx 2), b12 (idx 5)
    # R1 = a00 b02 -> C[0,2]
    U[0][col] = 1
    V[2][col] = 1
    W[2*2+0][col] = 1 # W[4]
    col += 1
    
    # R2 = a01 b12 -> C[0,2]
    U[1][col] = 1
    V[5][col] = 1
    W[4][col] = 1
    col += 1
    
    # R3 = a10 b02 -> C[1,2]
    U[2][col] = 1
    V[2][col] = 1
    W[2*2+1][col] = 1 # W[5]
    col += 1
    
    # R4 = a11 b12 -> C[1,2]
    U[3][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    # A_R (2x1) * B_RR (1x1)
    # B_RR: b22 (idx 8)
    # S1 = a02 b22 -> C[0,2]
    U[4][col] = 1
    V[8][col] = 1
    W[4][col] = 1
    col += 1
    
    # S2 = a12 b22 -> C[1,2]
    U[5][col] = 1
    V[8][col] = 1
    W[5][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
