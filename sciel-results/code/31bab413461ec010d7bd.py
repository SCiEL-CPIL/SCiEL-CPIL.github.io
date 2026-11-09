import time
import sys

DIMENSIONS = [2, 4, 8]

def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7) -> U[i*2+j, c]
    V: (4, 7) -> V[j*2+k, c]
    W: (4, 7) -> W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen products M1..M7
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A01 - A00)(B00 + B01)
    # M7 = (A10 - A11)(B10 + B11)
    
    # Indices: A[i,j], B[j,k], C[i,k]
    # U[i*2+j, c]
    # V[j*2+k, c]
    # W[k*2+i, c]
    
    # c=0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # C00 += M1, C11 += M1
    # W[k*2+i, 0]
    # C00 is k=0, i=0 -> W[0,0]=1
    # C11 is k=1, i=1 -> W[3,0]=1
    # Wait, W[k*2+i, c].
    # C[i,k].
    # M1 contributes to C00 and C11.
    # C00: i=0, k=0. W[0*2+0, 0] = W[0,0] = 1.
    # C11: i=1, k=1. W[1*2+1, 0] = W[3,0] = 1.
    
    # c=1: M2 = (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1 # C10: i=1, k=0 -> W[1*2+0, 1] = W[2,1] = 1
    # C10 += M2
    
    # c=2: M3 = A00(B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1 # C01: i=0, k=1 -> W[0*2+1, 2] = W[1,2] = 1
    # C01 += M3
    
    # c=3: M4 = A11(B10 - B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1 # C00 += M4, C10 += M4
    # C00: i=0, k=0 -> W[0,3] = 1
    # C10: i=1, k=0 -> W[2,3] = 1
    
    # c=4: M5 = (A00 + A10)B11
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1 # C00 -= M5, C01 += M5
    # C00: i=0, k=0 -> W[0,4] = -1
    # C01: i=0, k=1 -> W[1,4] = 1
    
    # c=5: M6 = (A01 - A00)(B00 + B01)
    U[1][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1 # C11 += M6
    # C11: i=1, k=1 -> W[3,5] = 1
    
    # c=6: M7 = (A10 - A11)(B10 + B11)
    U[2][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1 # C00 += M7
    # C00: i=0, k=0 -> W[0,6] = 1
    
    return U, V, W

def _combine_222_to_224(U1, V1, W1, U2, V2, W2):
    """
    Combines two 2x2x2 decompositions into a 2x2x4 decomposition.
    C_L = A B_L, C_R = A B_R.
    U is shared (same A).
    V and W are split by k.
    """
    rank1 = len(V1[0])
    rank2 = len(V2[0])
    total_rank = rank1 + rank2
    
    U_new = [[0]*total_rank for _ in range(4)]
    V_new = [[0]*total_rank for _ in range(8)] # 2*4
    W_new = [[0]*total_rank for _ in range(8)] # 4*2
    
    # U is concatenated
    for i in range(4):
        for c in range(rank1):
            U_new[i][c] = U1[i][c]
        for c in range(rank2):
            U_new[i][c + rank1] = U2[i][c]
            
    # V: j in 0..1, k in 0..3
    for j in range(2):
        for k in range(4):
            if k < 2:
                for c in range(rank1):
                    V_new[j*4 + k][c] = V1[j*2 + k][c]
            else:
                for c in range(rank2):
                    V_new[j*4 + k][c + rank1] = V2[j*2 + (k-2)][c]
                    
    # W: k in 0..3, i in 0..1
    for k in range(4):
        for i in range(2):
            if k < 2:
                for c in range(rank1):
                    W_new[k*2 + i][c] = W1[k*2 + i][c]
            else:
                for c in range(rank2):
                    W_new[k*2 + i][c + rank1] = W2[(k-2)*2 + i][c]
                    
    return U_new, V_new, W_new

def _combine_224_to_244(U1, V1, W1, U2, V2, W2):
    """
    Combines two 2x2x4 decompositions into a 2x4x4 decomposition.
    C = A_L B_T + A_R B_B.
    U is split by j.
    V is split by j.
    W is concatenated.
    """
    rank1 = len(V1[0])
    rank2 = len(V2[0])
    total_rank = rank1 + rank2
    
    U_new = [[0]*total_rank for _ in range(8)] # 2*4
    V_new = [[0]*total_rank for _ in range(16)] # 4*4
    W_new = [[0]*total_rank for _ in range(8)] # 4*2
    
    # U: i in 0..1, j in 0..3
    for i in range(2):
        for j in range(4):
            if j < 2:
                for c in range(rank1):
                    U_new[i*4 + j][c] = U1[i*2 + j][c]
            else:
                for c in range(rank2):
                    U_new[i*4 + j][c + rank1] = U2[i*2 + (j-2)][c]
                    
    # V: j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            if j < 2:
                for c in range(rank1):
                    V_new[j*4 + k][c] = V1[j*4 + k][c]
            else:
                for c in range(rank2):
                    V_new[j*4 + k][c + rank1] = V2[(j-2)*4 + k][c]
                    
    # W: k in 0..3, i in 0..1
    for k in range(4):
        for i in range(2):
            for c in range(rank1):
                W_new[k*2 + i][c] = W1[k*2 + i][c]
            for c in range(rank2):
                W_new[k*2 + i][c + rank1] = W2[k*2 + i][c]
                
    return U_new, V_new, W_new

def _combine_244_to_248(U1, V1, W1, U2, V2, W2):
    """
    Combines two 2x4x4 decompositions into a 2x4x8 decomposition.
    C_L = A B_L, C_R = A B_R.
    U is shared.
    V is split by k.
    W is split by k.
    """
    rank1 = len(V1[0])
    rank2 = len(V2[0])
    total_rank = rank1 + rank2
    
    U_new = [[0]*total_rank for _ in range(8)] # 2*4
    V_new = [[0]*total_rank for _ in range(32)] # 4*8
    W_new = [[0]*total_rank for _ in range(16)] # 8*2
    
    # U is concatenated
    for i in range(8):
        for c in range(rank1):
            U_new[i][c] = U1[i][c]
        for c in range(rank2):
            U_new[i][c + rank1] = U2[i][c]
            
    # V: j in 0..3, k in 0..7
    for j in range(4):
        for k in range(8):
            if k < 4:
                for c in range(rank1):
                    V_new[j*8 + k][c] = V1[j*4 + k][c]
            else:
                for c in range(rank2):
                    V_new[j*8 + k][c + rank1] = V2[j*4 + (k-4)][c]
                    
    # W: k in 0..7, i in 0..1
    for k in range(8):
        for i in range(2):
            if k < 4:
                for c in range(rank1):
                    W_new[k*2 + i][c] = W1[k*2 + i][c]
            else:
                for c in range(rank2):
                    W_new[k*2 + i][c + rank1] = W2[(k-4)*2 + i][c]
                    
    return U_new, V_new, W_new

def construct():
    start_time = time.time()
    
    # 1. Base: 2x2x2
    U1, V1, W1 = _strassen_222()
    
    # 2. Build 2x2x4 (two 2x2x2)
    U_224, V_224, W_224 = _combine_222_to_224(U1, V1, W1, U1, V1, W1)
    
    # 3. Build 2x4x4 (two 2x2x4)
    # Note: The two 2x2x4 blocks are identical in structure but applied to different parts of A and B.
    # U1, V1, W1 from step 2 are the same for both.
    U_244, V_244, W_244 = _combine_224_to_244(U_224, V_224, W_224, U_224, V_224, W_224)
    
    # 4. Build 2x4x8 (two 2x4x4)
    U_248, V_248, W_248 = _combine_244_to_248(U_244, V_244, W_244, U_244, V_244, W_244)
    
    rank = len(V_248[0])
    
    # Check time
    if time.time() - start_time > 140:
        # Fallback to schoolbook if something went wrong or took too long (unlikely here)
        pass
        
    return {"U": U_248, "V": V_248, "W": W_248}

