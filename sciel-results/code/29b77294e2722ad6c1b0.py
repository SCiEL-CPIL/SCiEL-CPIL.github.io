"""Rewritten program: structured algebraic decomposition of 2x2x15 tensor."""
import time

DIMENSIONS = [2, 2, 15]

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2.
    
    U shape: (4, 7)  indexed by [i*2+j]
    V shape: (4, 7)  indexed by [j*2+k]
    W shape: (4, 7)  indexed by [k*2+i]  (trace convention W[k*n+i, r])
    """
    # Strassen's 7 products for C = A*B where A,B are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U[i*2+j, r] = coefficient of A[i][j] in M_r
    # V[j*2+k, r] = coefficient of B[j][k] in M_r
    # W[k*2+i, r] = coefficient of M_r in C[i][k]  (trace convention: W[k*n+i])
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C00 += M1, C11 += M1
    W[0*2+0][0] = 1   # k=0,i=0 -> C[0][0]
    W[1*2+1][0] = 1   # k=1,i=1 -> C[1][1]
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C10 += M2, C11 -= M2
    W[0*2+1][1] = 1   # k=0,i=1 -> C[1][0]
    W[1*2+1][1] = -1  # k=1,i=1 -> C[1][1]
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C01 += M3, C11 += M3
    W[1*2+0][2] = 1   # k=1,i=0 -> C[0][1]
    W[1*2+1][2] = 1   # k=1,i=1 -> C[1][1]
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C00 += M4, C10 += M4
    W[0*2+0][3] = 1   # k=0,i=0
    W[0*2+1][3] = 1   # k=0,i=1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C00 -= M5, C01 += M5
    W[0*2+0][4] = -1  # k=0,i=0
    W[1*2+0][4] = 1   # k=1,i=0
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C11 += M6
    W[1*2+1][5] = 1   # k=1,i=1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C00 += M7
    W[0*2+0][6] = 1   # k=0,i=0
    
    return U, V, W

def _schoolbook_2x2x1():
    """Return rank-4 schoolbook decomposition for 2x2 * 2x1.
    
    U shape: (4, 4)
    V shape: (2, 4)   (m*p = 2*1 = 2)
    W shape: (2, 4)   (p*n = 1*2 = 2)
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(1):
                U[i*2+j][col] = 1
                V[j*1+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return U, V, W

def _schoolbook_2x1x1():
    """Return rank-2 schoolbook decomposition for 2x1 * 1x1.
    
    U shape: (2, 2)
    V shape: (1, 2)
    W shape: (2, 2)
    """
    U = [[0]*2 for _ in range(2)]
    V = [[0]*2 for _ in range(1)]
    W = [[0]*2 for _ in range(2)]
    
    col = 0
    for i in range(2):
        for j in range(1):
            for k in range(1):
                U[i*1+j][col] = 1
                V[j*1+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return U, V, W

def _block_2x2x8():
    """Build 2x2 * 2x8 decomposition using Strassen on four 2x2x2 blocks.
    
    Total rank: 4 * 7 = 28
    U shape: (4, 28)
    V shape: (16, 28)  (m*p = 2*8 = 16)
    W shape: (16, 28)  (p*n = 8*2 = 16)
    """
    sU, sV, sW = _strassen_2x2x2()
    rank = 28
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for block in range(4):
        base = block * 7
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*2+j][base+r] = sU[i*2+j][r]
                for k in range(2):
                    V[j*8 + block*2 + k][base+r] = sV[j*2+k][r]
                    W[(block*2+k)*2 + i][base+r] = sW[k*2+i][r]
    
    return U, V, W

def _block_2x2x7():
    """Build 2x2 * 2x7 decomposition.
    
    Split into three 2x2x2 blocks (rank 7 each) + one 2x2x1 block (rank 4).
    Total rank: 3*7 + 4 = 25
    U shape: (4, 25)
    V shape: (14, 25)  (m*p = 2*7 = 14)
    W shape: (14, 25)  (p*n = 7*2 = 14)
    """
    sU, sV, sW = _strassen_2x2x2()
    scU, scV, scW = _schoolbook_2x2x1()
    
    rank = 3*7 + 4
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(14)]
    W = [[0]*rank for _ in range(14)]
    
    # Three Strassen blocks for columns 0-5
    for block in range(3):
        base = block * 7
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*2+j][base+r] = sU[i*2+j][r]
                for k in range(2):
                    V[j*7 + block*2 + k][base+r] = sV[j*2+k][r]
                    W[(block*2+k)*2 + i][base+r] = sW[k*2+i][r]
    
    # One schoolbook 2x2x1 block for column 6
    base = 3*7
    for r in range(4):
        for i in range(2):
            for j in range(2):
                U[i*2+j][base+r] = scU[i*2+j][r]
        for k in range(1):
            for j in range(2):
                V[j*7 + 6][base+r] = scV[j*1+k][r]
            W[6*2 + i][base+r] = scW[k*2+i][r]
    
    return U, V, W

def construct():
    """Construct rank-minimal exact tensor decomposition for 2x2 * 2x15.
    
    Split p=15 into p1=8 and p2=7.
    Block 1: 2x2 * 2x8, rank 28 (4 Strassen 2x2x2)
    Block 2: 2x2 * 2x7, rank 25 (3 Strassen 2x2x2 + 1 schoolbook 2x2x1)
    Total rank: 53
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Build block 1: 2x2 * 2x8
    U1, V1, W1 = _block_2x2x8()
    
    if time.time() > deadline:
        # Fallback: return what we can
        pass
    
    # Build block 2: 2x2 * 2x7
    U2, V2, W2 = _block_2x2x7()
    
    if time.time() > deadline:
        pass
    
    # Combine blocks
    total_rank = len(U1[0]) + len(U2[0])
    
    U = [[0]*total_rank for _ in range(4)]
    V = [[0]*total_rank for _ in range(30)]  # m*p = 2*15 = 30
    W = [[0]*total_rank for _ in range(30)]  # p*n = 15*2 = 30
    
    r1 = len(U1[0])
    r2 = len(U2[0])
    
    # Copy block 1
    for i in range(4):
        for r in range(r1):
            U[i][r] = U1[i][r]
    for j in range(2):
        for k in range(8):
            for r in range(r1):
                V[j*15 + k][r] = V1[j*8 + k][r]
    for k in range(8):
        for i in range(2):
            for r in range(r1):
                W[k*2 + i][r] = W1[k*2 + i][r]
    
    # Copy block 2 (offset by r1)
    for i in range(4):
        for r in range(r2):
            U[i][r1 + r] = U2[i][r]
    for j in range(2):
        for k in range(7):
            for r in range(r2):
                V[j*15 + 8 + k][r1 + r] = V2[j*7 + k][r]
    for k in range(7):
        for i in range(2):
            for r in range(r2):
                W[(8+k)*2 + i][r1 + r] = W2[k*2 + i][r]
    
    return {"U": U, "V": V, "W": W}
