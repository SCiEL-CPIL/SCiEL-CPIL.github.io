"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check exact decomposition:
    sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] == delta(j1, j2)
    for all i, k, j1, j2
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0
                    for r in range(rank):
                        s += W[k*n+i][r] * U[i*m+j1][r] * V[j2*p+k][r]
                    target = 1 if j1 == j2 else 0
                    if abs(s - target) > 1e-9:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    # Initialize U, V, W as lists of lists (or numpy arrays converted to lists)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Strassen Terms (0-6) ---
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10 + A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3 = A00(B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[2][2] = 1; W[3][2] = -1
    
    # M4 = A11(B10 - B00)
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    # --- Schoolbook Terms for A[:,2] B[2,:] (7-12) ---
    # A02 B20 -> C00
    U[2][7] = 1; V[6][7] = 1; W[0][7] = 1
    # A02 B21 -> C01
    U[2][8] = 1; V[7][8] = 1; W[2][8] = 1
    # A02 B22 -> C02
    U[2][9] = 1; V[8][9] = 1; W[4][9] = 1
    # A12 B20 -> C10
    U[5][10] = 1; V[6][10] = 1; W[1][10] = 1
    # A12 B21 -> C11
    U[5][11] = 1; V[7][11] = 1; W[3][11] = 1
    # A12 B22 -> C12
    U[5][12] = 1; V[8][12] = 1; W[5][12] = 1
    
    # --- Schoolbook Terms for A[0:2,0:2] B[0:2,2] (13-16) ---
    # A00 B02 -> C02
    U[0][13] = 1; V[2][13] = 1; W[4][13] = 1
    # A01 B12 -> C02
    U[1][14] = 1; V[5][14] = 1; W[4][14] = 1
    # A10 B02 -> C12
    U[3][15] = 1; V[2][15] = 1; W[5][15] = 1
    # A11 B12 -> C12
    U[4][16] = 1; V[5][16] = 1; W[5][16] = 1
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: Schoolbook Rank 18
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
