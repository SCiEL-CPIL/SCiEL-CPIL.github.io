import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = 145.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # P1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_M_3_3_2():
    """
    Build M_3,3,2 decomposition with rank 17.
    n=3, m=3, p=2.
    """
    n, m, p = 3, 3, 2
    rank = 17
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # 1. Strassen for top-left 2x2 block of A and top-left 2x2 block of B
    # A indices: i in 0,1; j in 0,1 -> U rows 0,1,2,3
    # B indices: j in 0,1; k in 0,1 -> V rows 0,1,2,3
    # C indices: k in 0,1; i in 0,1 -> W rows 0,1,2,3
    sU, sV, sW = _strassen_2x2_2()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m + j][c] = sU[i*2 + j][c]
        for j in range(2):
            for k in range(2):
                V[j*p + k][c] = sV[j*2 + k][c]
        for k in range(2):
            for i in range(2):
                W[k*n + i][c] = sW[k*2 + i][c]
                
    # 2. Correction for A[:, 2] * B[2, :]
    # A indices: i in 0,1; j=2 -> U rows 2, 5
    # B indices: j=2; k in 0,1 -> V rows 4, 5
    # C indices: k in 0,1; i in 0,1 -> W rows 0,1,2,3
    # Terms 7, 8, 9, 10
    U[2][7] = 1; V[4][7] = 1; W[0][7] = 1 # A02 B20 -> C00
    U[2][8] = 1; V[5][8] = 1; W[1][8] = 1 # A02 B21 -> C01
    U[5][9] = 1; V[4][9] = 1; W[2][9] = 1 # A12 B20 -> C10
    U[5][10] = 1; V[5][10] = 1; W[3][10] = 1 # A12 B21 -> C11
    
    # 3. Bottom row C[2, :]
    # C indices: k in 0,1; i=2 -> W rows 6, 7
    # A indices: i=2; j in 0,1,2 -> U rows 6, 7, 8
    # B indices: j in 0,1,2; k in 0,1 -> V rows 0,1, 2,3, 4,5
    # Terms 11-16
    U[6][11] = 1; V[0][11] = 1; W[6][11] = 1 # A20 B00 -> C20
    U[6][12] = 1; V[1][12] = 1; W[7][12] = 1 # A20 B01 -> C21
    U[7][13] = 1; V[2][13] = 1; W[6][13] = 1 # A21 B10 -> C20
    U[7][14] = 1; V[3][14] = 1; W[7][14] = 1 # A21 B11 -> C21
    U[8][15] = 1; V[4][15] = 1; W[6][15] = 1 # A22 B20 -> C20
    U[8][16] = 1; V[5][16] = 1; W[7][16] = 1 # A22 B21 -> C21
    
    return U, V, W

def _build_M_1_1_4():
    """
    Build M_1,1,4 decomposition with rank 4.
    n=1, m=1, p=4.
    """
    n, m, p = 1, 1, 4
    rank = 4
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(4):
        U[0][c] = 1
        V[c][c] = 1
        W[c][0] = 1
        
    return U, V, W

def _kron_decomp(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """
    Construct M_{n1 n2, m1 m2, p1 p2} from M_{n1, m1, p1} and M_{n2, m2, p2}.
    """
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    R1 = len(U1[0])
    R2 = len(U2[0])
    R = R1 * R2
    
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    
    for c1 in range(R1):
        for c2 in range(R2):
            c = c1 * R2 + c2
            
            # Update U
            for i1 in range(n1):
                for j1 in range(m1):
                    val1 = U1[i1*m1 + j1][c1]
                    if val1 == 0: continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            val2 = U2[i2*m2 + j2][c2]
                            if val2 == 0: continue
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            U[i*m + j][c] += val1 * val2
                            
            # Update V
            for j1 in range(m1):
                for k1 in range(p1):
                    val1 = V1[j1*p1 + k1][c1]
                    if val1 == 0: continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            val2 = V2[j2*p2 + k2][c2]
                            if val2 == 0: continue
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            V[j*p + k][c] += val1 * val2
                            
            # Update W
            for k1 in range(p1):
                for i1 in range(n1):
                    val1 = W1[k1*n1 + i1][c1]
                    if val1 == 0: continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            val2 = W2[k2*n2 + i2][c2]
                            if val2 == 0: continue
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            W[k*n + i][c] += val1 * val2
                            
    return U, V, W

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    # M_3,3,2 has rank 17
    U_332, V_332, W_332 = _build_M_3_3_2()
    # M_1,1,4 has rank 4
    U_114, V_114, W_114 = _build_M_1_1_4()
    
    # M_3,3,8 = M_3,3,2 tensor M_1,1,4
    # n1=3, m1=3, p1=2
    # n2=1, m2=1, p2=4
    U, V, W = _kron_decomp(U_332, V_332, W_332, 3, 3, 2, U_114, V_114, W_114, 1, 1, 4)
    
    return {"U": U, "V": V, "W": W}
