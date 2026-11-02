import time

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    # Rank 11 decomposition: Strassen for first 2 cols, Schoolbook for 3rd col
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen terms for C[:, 0:2]
    # M1 = (A00+A11)(B00+B11)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[4][0] += 1
    W[0][0] += 1; W[2][0] += 1 # C00, C01
    
    # M2 = (A10+A11)B00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1; W[3][1] += 1 # C10, C11
    
    # M3 = A00(B01-B11)
    U[0][2] += 1
    V[1][2] += 1; V[4][2] -= 1
    W[2][2] += 1; W[3][2] += 1 # C01, C11
    
    # M4 = A11(B10-B00)
    U[3][3] += 1
    V[3][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[1][3] += 1 # C00, C10
    
    # M5 = (A00+A01)B11
    U[0][4] += 1; U[1][4] += 1
    V[4][4] += 1
    W[0][4] -= 1; W[2][4] += 1 # C00, C01
    
    # M6 = (A01-A00)(B00+B10)
    U[1][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[3][5] += 1
    W[2][5] -= 1; W[3][5] += 1 # C01, C11
    
    # M7 = (A10-A11)(B11+B01)
    U[2][6] += 1; U[3][6] -= 1
    V[4][6] += 1; V[1][6] += 1
    W[0][6] += 1; W[1][6] -= 1 # C00, C10
    
    # Schoolbook terms for C[:, 2]
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # Term 7 (index 7): A00*B02
    U[0][7] += 1
    V[2][7] += 1
    W[4][7] += 1 # C02
    
    # Term 8 (index 8): A01*B12
    U[1][8] += 1
    V[5][8] += 1
    W[4][8] += 1 # C02
    
    # Term 9 (index 9): A10*B02
    U[2][9] += 1
    V[2][9] += 1
    W[5][9] += 1 # C12
    
    # Term 10 (index 10): A11*B12
    U[3][10] += 1
    V[5][10] += 1
    W[5][10] += 1 # C12
    
    return {"U": U, "V": V, "W": W}
