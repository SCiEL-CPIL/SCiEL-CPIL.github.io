import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # Rank 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Results
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M6
    
    # Define blocks
    # Block 0: A11 B11 -> C00, C01 (cols 0,1)
    # Block 1: A12 B21 -> C00, C01 (cols 0,1)
    # Block 2: A11 B12 -> C00, C01 (cols 2,3)
    # Block 3: A12 B22 -> C00, C01 (cols 2,3)
    
    # Indices in A: i in 0..1, j in 0..3
    # Indices in B: j in 0..3, k in 0..3
    # Indices in C: i in 0..1, k in 0..3
    
    # Helper to add a Strassen block
    # a_offset_i, a_offset_j: offset for A indices
    # b_offset_j, b_offset_k: offset for B indices
    # c_offset_i, c_offset_k: offset for C indices
    # rank_offset: starting column in U, V, W
    
    def add_strassen_block(a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, r_off):
        # M1
        U[a_i_off*4 + a_j_off + 0][r_off + 0] += 1 # A00
        U[a_i_off*4 + a_j_off + 1][r_off + 0] += 1 # A11 (j+1)
        V[(b_j_off + 0)*4 + b_k_off + 0][r_off + 0] += 1 # B00
        V[(b_j_off + 1)*4 + b_k_off + 1][r_off + 0] += 1 # B11 (j+1, k+1)
        # Wait, B is 4x4. B[j,k].
        # B00 is B[b_j_off, b_k_off]
        # B11 is B[b_j_off+1, b_k_off+1]
        # So V index is (b_j_off)*4 + (b_k_off) for B00.
        # V index is (b_j_off+1)*4 + (b_k_off+1) for B11.
        
        # Let's rewrite V indices carefully.
        # V row index = j*4 + k.
        
        # M1
        U[a_i_off*4 + (a_j_off + 0)][r_off + 0] = 1
        U[a_i_off*4 + (a_j_off + 1)][r_off + 0] = 1
        V[(b_j_off + 0)*4 + (b_k_off + 0)][r_off + 0] = 1
        V[(b_j_off + 1)*4 + (b_k_off + 1)][r_off + 0] = 1
        W[(c_k_off + 0)*2 + (c_i_off + 0)][r_off + 0] = 1 # C00
        W[(c_k_off + 1)*2 + (c_i_off + 1)][r_off + 0] = 1 # C11
        
        # M2
        U[a_i_off*4 + (a_j_off + 1)][r_off + 1] = 1 # A10 (i+1, j+0) -> A[i+1, j]
        # Wait, A10 is A[1,0] in local 2x2.
        # Local A[1,0] is global A[a_i_off+1, a_j_off+0].
        # Local A[1,1] is global A[a_i_off+1, a_j_off+1].
        U[a_i_off*4 + (a_j_off + 0)][r_off + 1] = 0
        U[a_i_off*4 + (a_j_off + 1)][r_off + 1] = 1
        # M2 = (A10 + A11) B00
        U[a_i_off*4 + (a_j_off + 0)][r_off + 1] = 0 # A10 is local [1,0]
        # Correction: Local A[1,0] is global A[a_i_off+1, a_j_off+0].
        # Local A[1,1] is global A[a_i_off+1, a_j_off+1].
        U[a_i_off*4 + (a_j_off + 0)][r_off + 1] = 0
        U[a_i_off*4 + (a_j_off + 1)][r_off + 1] = 1
        # Wait, A10 is row 1, col 0. A11 is row 1, col 1.
        # So U[1,0] = 1, U[1,1] = 1.
        U[a_i_off*4 + (a_j_off + 0)][r_off + 1] = 0
        U[a_i_off*4 + (a_j_off + 1)][r_off + 1] = 1
        # This is getting messy. Let's use a helper for local indices.
        
        pass

    # Let's implement directly with explicit indices to avoid mistakes.
    
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # r = 0..6
    # M1
    U[0][0] = 1; U[1][0] = 1
    V[0][0] = 1; V[5][0] = 1 # B[0,0]=V[0], B[1,1]=V[1*4+1=5]
    W[0][0] = 1; W[3][0] = 1 # C[0,0]=W[0], C[1,1]=W[1*2+1=3]
    
    # M2
    U[1][1] = 1; U[2][1] = 1 # A[1,0]=U[1], A[1,1]=U[2]
    V[0][1] = 1 # B[0,0]
    W[2][1] = 1 # C[1,0]=W[0*2+1=1? No. C[1,0] is i=1, k=0. W[k*n+i] = W[0*2+1]=W[1].
    # Wait, C[1,0] means row 1, col 0.
    # W index is k*n + i. k=0, i=1 -> W[1].
    # C[0,1] means row 0, col 1. k=1, i=0 -> W[2].
    # C[1,1] means row 1, col 1. k=1, i=1 -> W[3].
    # C[0,0] means row 0, col 0. k=0, i=0 -> W[0].
    
    # Let's re-verify W indices.
    # W[k*n+i, r]. n=2.
    # C[i,k].
    # C[0,0] -> k=0, i=0 -> W[0]
    # C[0,1] -> k=1, i=0 -> W[2]
    # C[1,0] -> k=0, i=1 -> W[1]
    # C[1,1] -> k=1, i=1 -> W[3]
    
    # M1 contributes to C[0,0] and C[1,1].
    # C[0,0] coeff 1. C[1,1] coeff 1.
    # So W[0][0] = 1, W[3][0] = 1. Correct.
    
    # M2 contributes to C[1,0].
    # C[1,0] coeff 1.
    # So W[1][1] = 1.
    
    # M3 = A[0,0](B[1,0]-B[0,0])
    U[0][2] = 1
    V[4][2] = 1 # B[1,0] = V[1*4+0]=4
    V[0][2] = -1 # B[0,0]
    # M3 contributes to C[0,1] and C[1,1] with -1.
    # C[0,1] coeff 1. C[1,1] coeff -1.
    W[2][2] = 1
    W[3][2] = -1
    
    # M4 = A[1,1](B[0,1]-B[1,1])
    U[2][3] = 1
    V[1][3] = 1 # B[0,1] = V[0*4+1]=1
    V[5][3] = -1 # B[1,1]
    # M4 contributes to C[0,0] and C[1,0].
    # C[0,0] coeff 1. C[1,0] coeff 1.
    W[0][3] = 1
    W[1][3] = 1
    
    # M5 = (A[0,0]+A[0,1])B[1,1]
    U[0][4] = 1
    U[1][4] = 1
    V[5][4] = 1
    # M5 contributes to C[0,0] with -1 and C[0,1] with 1.
    W[0][4] = -1
    W[2][4] = 1
    
    # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    U[1][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    # M6 contributes to C[1,1] with -1.
    W[3][5] = -1
    
    # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
    U[1][6] = 1
    U[2][6] = -1
    V[4][6] = 1
    V[5][6] = 1
    # M7 contributes to C[0,0] with 1.
    W[0][6] = 1
    
    # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # r = 7..13
    # A indices: i in 0,1; j in 2,3
    # B indices: j in 2,3; k in 0,1
    # C indices: i in 0,1; k in 0,1
    
    # M1
    U[0+2][7] = 1; U[1+2][7] = 1 # A[0,2], A[1,3]
    V[2*4+0][7] = 1; V[3*4+1][7] = 1 # B[2,0], B[3,1]
    W[0][7] = 1; W[3][7] = 1
    
    # M2
    U[1+2][8] = 1; U[1+3][8] = 1 # A[1,2], A[1,3]
    V[2*4+0][8] = 1 # B[2,0]
    W[1][8] = 1
    
    # M3
    U[0+2][9] = 1 # A[0,2]
    V[3*4+0][9] = 1 # B[3,0]
    V[2*4+0][9] = -1 # B[2,0]
    W[2][9] = 1
    W[3][9] = -1
    
    # M4
    U[1+3][10] = 1 # A[1,3]
    V[2*4+1][10] = 1 # B[2,1]
    V[3*4+1][10] = -1 # B[3,1]
    W[0][10] = 1
    W[1][10] = 1
    
    # M5
    U[0+2][11] = 1
    U[0+3][11] = 1
    V[3*4+1][11] = 1
    W[0][11] = -1
    W[2][11] = 1
    
    # M6
    U[1+2][12] = 1
    U[0+2][12] = -1
    V[2*4+0][12] = 1
    V[2*4+1][12] = 1
    W[3][12] = -1
    
    # M7
    U[0+3][13] = 1
    U[1+3][13] = -1
    V[3*4+0][13] = 1
    V[3*4+1][13] = 1
    W[0][13] = 1

    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # r = 14..20
    # A indices: i in 0,1; j in 0,1
    # B indices: j in 0,1; k in 2,3
    # C indices: i in 0,1; k in 2,3
    
    # M1
    U[0][14] = 1; U[1][14] = 1
    V[0*4+2][14] = 1; V[1*4+3][14] = 1
    W[2*2+0][14] = 1; W[3*2+1][14] = 1 # C[0,2]=W[4], C[1,3]=W[7]
    
    # M2
    U[1][15] = 1; U[2][15] = 1
    V[0*4+2][15] = 1
    W[2*2+1][15] = 1 # C[1,2]=W[5]
    
    # M3
    U[0][16] = 1
    V[1*4+2][16] = 1
    V[0*4+2][16] = -1
    W[3*2+0][16] = 1 # C[0,3]=W[6]
    W[3*2+1][16] = -1 # C[1,3]=W[7]
    
    # M4
    U[2][17] = 1
    V[0*4+3][17] = 1
    V[1*4+3][17] = -1
    W[2*2+0][17] = 1
    W[2*2+1][17] = 1
    
    # M5
    U[0][18] = 1
    U[1][18] = 1
    V[1*4+3][18] = 1
    W[2*2+0][18] = -1
    W[3*2+0][18] = 1
    
    # M6
    U[1][19] = 1
    U[0][19] = -1
    V[0*4+2][19] = 1
    V[0*4+3][19] = 1
    W[3*2+1][19] = -1
    
    # M7
    U[1][20] = 1
    U[2][20] = -1
    V[1*4+2][20] = 1
    V[1*4+3][20] = 1
    W[2*2+0][20] = 1

    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # r = 21..27
    # A indices: i in 0,1; j in 2,3
    # B indices: j in 2,3; k in 2,3
    # C indices: i in 0,1; k in 2,3
    
    # M1
    U[0+2][21] = 1; U[1+2][21] = 1
    V[2*4+2][21] = 1; V[3*4+3][21] = 1
    W[2*2+0][21] = 1; W[3*2+1][21] = 1
    
    # M2
    U[1+2][22] = 1; U[1+3][22] = 1
    V[2*4+2][22] = 1
    W[2*2+1][22] = 1
    
    # M3
    U[0+2][23] = 1
    V[3*4+2][23] = 1
    V[2*4+2][23] = -1
    W[3*2+0][23] = 1
    W[3*2+1][23] = -1
    
    # M4
    U[1+3][24] = 1
    V[2*4+3][24] = 1
    V[3*4+3][24] = -1
    W[2*2+0][24] = 1
    W[2*2+1][24] = 1
    
    # M5
    U[0+2][25] = 1
    U[0+3][25] = 1
    V[3*4+3][25] = 1
    W[2*2+0][25] = -1
    W[3*2+0][25] = 1
    
    # M6
    U[1+2][26] = 1
    U[0+2][26] = -1
    V[2*4+2][26] = 1
    V[2*4+3][26] = 1
    W[3*2+1][26] = -1
    
    # M7
    U[0+3][27] = 1
    U[1+3][27] = -1
    V[3*4+2][27] = 1
    V[3*4+3][27] = 1
    W[2*2+0][27] = 1

    return {"U": U, "V": V, "W": W}
