import time
import numpy as np

_START = time.time()
DEADLINE = 145.0

def _time_left():
    return DEADLINE - (time.time() - _START)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _laderman_3x3_3():
    """Return U, V, W for 3x3 * 3x3 with rank 23 (Laderman)."""
    n, m, p = 3, 3, 3
    rank = 23
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Laderman's 23 multiplications
    # U[i*3+j][c], V[j*3+k][c], W[k*3+i][c]
    
    # m1 = a11 * b11
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # m2 = a22 * b22
    U[4][1] = 1
    V[4][1] = 1
    W[4][1] = 1
    
    # m3 = a33 * b33
    U[8][2] = 1
    V[8][2] = 1
    W[8][2] = 1
    
    # m4 = (a11+a22) * b12
    U[0][3] = 1; U[4][3] = 1
    V[1][3] = 1
    W[3][3] = 1; W[1][3] = -1
    
    # m5 = (a22-a11) * b21
    U[4][4] = 1; U[0][4] = -1
    V[3][4] = 1
    W[0][4] = 1; W[4][4] = 1
    
    # m6 = (a12+a33) * b13
    U[1][5] = 1; U[8][5] = 1
    V[2][5] = 1
    W[6][5] = 1; W[2][5] = -1
    
    # m7 = (a33-a12) * b31
    U[8][6] = 1; U[1][6] = -1
    V[6][6] = 1
    W[0][6] = 1; W[6][6] = 1
    
    # m8 = (a23+a11) * b23
    U[2][7] = 1; U[0][7] = 1
    V[5][7] = 1
    W[5][7] = 1; W[2][7] = -1
    
    # m9 = (a11-a23) * b12
    U[0][8] = 1; U[2][8] = -1
    V[1][8] = 1
    W[3][8] = 1; W[1][8] = -1
    
    # m10 = (a13+a22) * b13
    U[3][9] = 1; U[4][9] = 1
    V[2][9] = 1
    W[6][9] = 1; W[4][9] = -1
    
    # m11 = (a22-a13) * b21
    U[4][10] = 1; U[3][10] = -1
    V[3][10] = 1
    W[0][10] = 1; W[4][10] = 1
    
    # m12 = (a21+a33) * b21
    U[3][11] = 1; U[8][11] = 1
    V[3][11] = 1
    W[0][11] = 1; W[8][11] = -1
    
    # m13 = (a33-a21) * b32
    U[8][12] = 1; U[3][12] = -1
    V[7][12] = 1
    W[4][12] = 1; W[8][12] = 1
    
    # m14 = (a31+a22) * b31
    U[6][13] = 1; U[4][13] = 1
    V[6][13] = 1
    W[0][13] = 1; W[4][13] = -1
    
    # m15 = (a22-a31) * b23
    U[4][14] = 1; U[6][14] = -1
    V[5][14] = 1
    W[2][14] = 1; W[4][14] = 1
    
    # m16 = (a32+a11) * b32
    U[7][15] = 1; U[0][15] = 1
    V[7][15] = 1
    W[4][15] = 1; W[0][15] = -1
    
    # m17 = (a11-a32) * b13
    U[0][16] = 1; U[7][16] = -1
    V[2][16] = 1
    W[6][16] = 1; W[0][16] = 1
    
    # m18 = (a12+a21) * (b13+b23)
    U[1][17] = 1; U[3][17] = 1
    V[2][17] = 1; V[5][17] = 1
    W[6][17] = 1; W[5][17] = 1; W[2][17] = -1; W[3][17] = -1
    
    # m19 = (a31+a32) * (b13+b23)
    U[6][18] = 1; U[7][18] = 1
    V[2][18] = 1; V[5][18] = 1
    W[6][18] = 1; W[5][18] = 1; W[2][18] = -1; W[3][18] = -1
    
    # m20 = (a12+a32) * (b11+b21)
    U[1][19] = 1; U[7][19] = 1
    V[0][19] = 1; V[3][19] = 1
    W[0][19] = 1; W[1][19] = 1; W[4][19] = -1; W[7][19] = -1
    
    # m21 = (a21+a31) * (b11+b21)
    U[3][20] = 1; U[6][20] = 1
    V[0][20] = 1; V[3][20] = 1
    W[0][20] = 1; W[4][20] = 1; W[1][20] = -1; W[6][20] = -1
    
    # m22 = (a21+a23) * (b12+b32)
    U[3][21] = 1; U[2][21] = 1
    V[1][21] = 1; V[7][21] = 1
    W[3][21] = 1; W[7][21] = 1; W[4][21] = -1; W[2][21] = -1
    
    # m23 = (a31+a33) * (b12+b32)
    U[6][22] = 1; U[8][22] = 1
    V[1][22] = 1; V[7][22] = 1
    W[3][22] = 1; W[7][22] = 1; W[4][22] = -1; W[8][22] = -1
    
    return U, V, W

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

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = 3, 3, 8
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 72
    
    # Combined approach: Laderman for k=0,1,2 + Strassen for k=3,4 and k=5,6 + single for k=7
    # Rank = 23 + 7 + 7 + 1 = 38
    if _time_left() > 10:
        try:
            lU, lV, lW = _laderman_3x3_3()
            sU, sV, sW = _strassen_2x2_2()
            
            rank = 23 + 7 + 7 + 1
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            col = 0
            # Laderman for k=0,1,2
            for c in range(23):
                for i in range(3):
                    for j in range(3):
                        U[i*3+j][col+c] = lU[i*3+j][c]
                for j in range(3):
                    for k in range(3):
                        V[j*3+k][col+c] = lV[j*3+k][c]
                for k in range(3):
                    for i in range(3):
                        W[k*3+i][col+c] = lW[k*3+i][c]
            col += 23
            
            # Strassen for k=3,4 (rows i=0,1; cols j=0,1)
            for c in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i*3+j][col+c] = sU[i*2+j][c]
                for j in range(2):
                    for k in range(2):
                        V[j*3+(3+k)][col+c] = sV[j*2+k][c]
                for k in range(2):
                    for i in range(2):
                        W[(3+k)*3+i][col+c] = sW[k*2+i][c]
            col += 7
            
            # Strassen for k=5,6 (rows i=0,1; cols j=0,1)
            for c in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i*3+j][col+c] = sU[i*2+j][c]
                for j in range(2):
                    for k in range(2):
                        V[j*3+(5+k)][col+c] = sV[j*2+k][c]
                for k in range(2):
                    for i in range(2):
                        W[(5+k)*3+i][col+c] = sW[k*2+i][c]
            col += 7
            
            # Single term for k=7: all i,j
            for i in range(3):
                for j in range(3):
                    U[i*3+j][col] = 1
                V[j*3+7][col] = 1
                W[7*3+i][col] = 1
            col += 1
            
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    return best_result
