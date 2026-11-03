import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # M1: U[X00+1, X11+1], V[Y00+1, Y11+1], W[Z00+1, Z11+1]
    # M2: U[X10+1, X11+1], V[Y00+1], W[Z10+1, Z11-1]
    # M3: U[X00+1], V[Y01+1, Y11-1], W[Z01+1, Z11+1]
    # M4: U[X11+1], V[Y10+1, Y00-1], W[Z00+1, Z10+1]
    # M5: U[X00+1, X01+1], V[Y11+1], W[Z00-1, Z01+1]
    # M6: U[X10+1, X00-1], V[Y00+1, Y01+1], W[Z11+1]
    # M7: U[X01+1, X11-1], V[Y10+1, Y11+1], W[Z00+1, Z11+1]
    
    def add_strassen(col_start, u_idx, v_idx, w_idx):
        # u_idx, v_idx, w_idx are lists of 4 indices: [00, 01, 10, 11]
        
        # M1
        U[u_idx[0]][col_start] += 1
        U[u_idx[3]][col_start] += 1
        V[v_idx[0]][col_start] += 1
        V[v_idx[3]][col_start] += 1
        W[w_idx[0]][col_start] += 1
        W[w_idx[3]][col_start] += 1
        
        # M2
        U[u_idx[2]][col_start+1] += 1
        U[u_idx[3]][col_start+1] += 1
        V[v_idx[0]][col_start+1] += 1
        W[w_idx[2]][col_start+1] += 1
        W[w_idx[3]][col_start+1] -= 1
        
        # M3
        U[u_idx[0]][col_start+2] += 1
        V[v_idx[1]][col_start+2] += 1
        V[v_idx[3]][col_start+2] -= 1
        W[w_idx[1]][col_start+2] += 1
        W[w_idx[3]][col_start+2] += 1
        
        # M4
        U[u_idx[3]][col_start+3] += 1
        V[v_idx[2]][col_start+3] += 1
        V[v_idx[0]][col_start+3] -= 1
        W[w_idx[0]][col_start+3] += 1
        W[w_idx[2]][col_start+3] += 1
        
        # M5
        U[u_idx[0]][col_start+4] += 1
        U[u_idx[1]][col_start+4] += 1
        V[v_idx[3]][col_start+4] += 1
        W[w_idx[0]][col_start+4] -= 1
        W[w_idx[1]][col_start+4] += 1
        
        # M6
        U[u_idx[2]][col_start+5] += 1
        U[u_idx[0]][col_start+5] -= 1
        V[v_idx[0]][col_start+5] += 1
        V[v_idx[1]][col_start+5] += 1
        W[w_idx[3]][col_start+5] += 1
        
        # M7
        U[u_idx[1]][col_start+6] += 1
        U[u_idx[3]][col_start+6] -= 1
        V[v_idx[2]][col_start+6] += 1
        V[v_idx[3]][col_start+6] += 1
        W[w_idx[0]][col_start+6] += 1
        W[w_idx[3]][col_start+6] += 1

    # Define global indices for A, B, C blocks
    # A is 2x4. A1 (cols 0,1), A2 (cols 2,3)
    # A1 indices: 0,1,2,3 (row-major: 00, 01, 10, 11)
    # A2 indices: 4,5,6,7
    A1_idx = [0, 1, 2, 3]
    A2_idx = [4, 5, 6, 7]
    
    # B is 4x8.
    # B11 (rows 0,1; cols 0,1): 0,1,4,5
    # B12 (rows 0,1; cols 2,3): 2,3,6,7
    # B13 (rows 0,1; cols 4,5): 8,9,12,13
    # B14 (rows 0,1; cols 6,7): 10,11,14,15
    # B21 (rows 2,3; cols 0,1): 16,17,20,21
    # B22 (rows 2,3; cols 2,3): 18,19,22,23
    # B23 (rows 2,3; cols 4,5): 24,25,28,29
    # B24 (rows 2,3; cols 6,7): 26,27,30,31
    
    B11_idx = [0, 1, 4, 5]
    B12_idx = [2, 3, 6, 7]
    B13_idx = [8, 9, 12, 13]
    B14_idx = [10, 11, 14, 15]
    B21_idx = [16, 17, 20, 21]
    B22_idx = [18, 19, 22, 23]
    B23_idx = [24, 25, 28, 29]
    B24_idx = [26, 27, 30, 31]
    
    # C is 8x2.
    # C0 (cols 0,1; rows 0,1): 0,1,2,3
    # C1 (cols 2,3; rows 0,1): 4,5,6,7
    # C2 (cols 4,5; rows 0,1): 8,9,10,11
    # C3 (cols 6,7; rows 0,1): 12,13,14,15
    
    C0_idx = [0, 1, 2, 3]
    C1_idx = [4, 5, 6, 7]
    C2_idx = [8, 9, 10, 11]
    C3_idx = [12, 13, 14, 15]
    
    # C0 = A1*B11 + A2*B21
    # Term 1: A1*B11
    add_strassen(0, A1_idx, B11_idx, C0_idx)
    # Term 2: A2*B21
    add_strassen(7, A2_idx, B21_idx, C0_idx)
    
    # C1 = A1*B12 + A2*B22
    # Term 1: A1*B12
    add_strassen(14, A1_idx, B12_idx, C1_idx)
    # Term 2: A2*B22
    add_strassen(21, A2_idx, B22_idx, C1_idx)
    
    # C2 = A1*B13 + A2*B23
    # Term 1: A1*B13
    add_strassen(28, A1_idx, B13_idx, C2_idx)
    # Term 2: A2*B23
    add_strassen(35, A2_idx, B23_idx, C2_idx)
    
    # C3 = A1*B14 + A2*B24
    # Term 1: A1*B14
    add_strassen(42, A1_idx, B14_idx, C3_idx)
    # Term 2: A2*B24
    add_strassen(49, A2_idx, B24_idx, C3_idx)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
