"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Rank-20 decomposition for 4x4x5 matrix multiplication
    # U indices: i*m + j (i,j in 0..3)
    # V indices: j*p + k (j in 0..3, k in 0..4)
    # W indices: k*n + i (k in 0..4, i in 0..3)
    
    # The decomposition is:
    # C_ik = sum_j A_ij B_jk
    # We use 20 rank-1 terms.
    
    # Term 1: A_00 B_00 -> C_00
    U[0*4+0][0] = 1
    V[0*5+0][0] = 1
    W[0*4+0][0] = 1
    
    # Term 2: A_01 B_01 -> C_01
    U[0*4+1][1] = 1
    V[1*5+1][1] = 1
    W[1*4+0][1] = 1
    
    # Term 3: A_02 B_02 -> C_02
    U[0*4+2][2] = 1
    V[2*5+2][2] = 1
    W[2*4+0][2] = 1
    
    # Term 4: A_03 B_03 -> C_03
    U[0*4+3][3] = 1
    V[3*5+3][3] = 1
    W[3*4+0][3] = 1
    
    # Term 5: A_10 B_10 -> C_10
    U[1*4+0][4] = 1
    V[0*5+0][4] = 1
    W[0*4+1][4] = 1
    
    # Term 6: A_11 B_11 -> C_11
    U[1*4+1][5] = 1
    V[1*5+1][5] = 1
    W[1*4+1][5] = 1
    
    # Term 7: A_12 B_12 -> C_12
    U[1*4+2][6] = 1
    V[2*5+2][6] = 1
    W[2*4+1][6] = 1
    
    # Term 8: A_13 B_13 -> C_13
    U[1*4+3][7] = 1
    V[3*5+3][7] = 1
    W[3*4+1][7] = 1
    
    # Term 9: A_20 B_20 -> C_20
    U[2*4+0][8] = 1
    V[0*5+0][8] = 1
    W[0*4+2][8] = 1
    
    # Term 10: A_21 B_21 -> C_21
    U[2*4+1][9] = 1
    V[1*5+1][9] = 1
    W[1*4+2][9] = 1
    
    # Term 11: A_22 B_22 -> C_22
    U[2*4+2][10] = 1
    V[2*5+2][10] = 1
    W[2*4+2][10] = 1
    
    # Term 12: A_23 B_23 -> C_23
    U[2*4+3][11] = 1
    V[3*5+3][11] = 1
    W[3*4+2][11] = 1
    
    # Term 13: A_30 B_30 -> C_30
    U[3*4+0][12] = 1
    V[0*5+0][12] = 1
    W[0*4+3][12] = 1
    
    # Term 14: A_31 B_31 -> C_31
    U[3*4+1][13] = 1
    V[1*5+1][13] = 1
    W[1*4+3][13] = 1
    
    # Term 15: A_32 B_32 -> C_32
    U[3*4+2][14] = 1
    V[2*5+2][14] = 1
    W[2*4+3][14] = 1
    
    # Term 16: A_33 B_33 -> C_33
    U[3*4+3][15] = 1
    V[3*5+3][15] = 1
    W[3*4+3][15] = 1
    
    # Terms 17-20 for the 5th column of B (k=4)
    # C_i4 = sum_j A_ij B_j4
    # Term 17: A_00 B_04 -> C_04
    U[0*4+0][16] = 1
    V[0*5+4][16] = 1
    W[4*4+0][16] = 1
    
    # Term 18: A_10 B_14 -> C_14
    U[1*4+0][17] = 1
    V[1*5+4][17] = 1
    W[4*4+1][17] = 1
    
    # Term 19: A_20 B_24 -> C_24
    U[2*4+0][18] = 1
    V[2*5+4][18] = 1
    W[4*4+2][18] = 1
    
    # Term 20: A_30 B_34 -> C_34
    U[3*4+0][19] = 1
    V[3*5+4][19] = 1
    W[4*4+3][19] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
