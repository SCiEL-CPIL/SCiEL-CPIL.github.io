import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = 28  # 4 blocks of 2x2x2, each rank 7
    
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (32, 28)
    W = [[0] * rank for _ in range(p * n)]  # (16, 28)
    
    # Strassen rank-7 decomposition for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    col = 0
    for x in range(4):
        for r in range(7):
            # U contributions (from A_{i, 2a+b})
            if r == 0:
                U[0 * m + 0][col] = 1   # A_{0,0}
                U[1 * m + 3][col] = 1   # A_{1,3}
            elif r == 1:
                U[1 * m + 2][col] = 1   # A_{1,2}
                U[1 * m + 3][col] = 1   # A_{1,3}
            elif r == 2:
                U[0 * m + 0][col] = 1   # A_{0,0}
            elif r == 3:
                U[1 * m + 3][col] = 1   # A_{1,3}
            elif r == 4:
                U[0 * m + 0][col] = 1   # A_{0,0}
                U[0 * m + 1][col] = 1   # A_{0,1}
            elif r == 5:
                U[1 * m + 2][col] = 1   # A_{1,2}
                U[0 * m + 0][col] = -1  # A_{0,0}
            elif r == 6:
                U[0 * m + 1][col] = 1   # A_{0,1}
                U[1 * m + 3][col] = -1  # A_{1,3}
            
            # V contributions (from B_{2a+b, 2x+j})
            # V index = j_row * p + k_col
            if r == 0:
                V[0 * p + (2*x + 0)][col] = 1  # B_{0, 2x+0}
                V[3 * p + (2*x + 1)][col] = 1  # B_{3, 2x+1}
            elif r == 1:
                V[0 * p + (2*x + 0)][col] = 1  # B_{0, 2x+0}
            elif r == 2:
                V[2 * p + (2*x + 0)][col] = 1  # B_{2, 2x+0}
                V[3 * p + (2*x + 1)][col] = -1 # B_{3, 2x+1}
            elif r == 3:
                V[1 * p + (2*x + 1)][col] = 1  # B_{1, 2x+1}
                V[2 * p + (2*x + 0)][col] = -1 # B_{2, 2x+0}
            elif r == 4:
                V[3 * p + (2*x + 1)][col] = 1  # B_{3, 2x+1}
            elif r == 5:
                V[0 * p + (2*x + 0)][col] = 1  # B_{0, 2x+0}
                V[2 * p + (2*x + 0)][col] = 1  # B_{2, 2x+0}
            elif r == 6:
                V[1 * p + (2*x + 1)][col] = 1  # B_{1, 2x+1}
                V[3 * p + (2*x + 1)][col] = 1  # B_{3, 2x+1}
            
            # W contributions (to C_{i, 2x+j}), W index = (2x+j)*n + i
            if r == 0:
                W[(2*x + 0) * n + 0][col] = 1  # C_{0, 2x+0}
                W[(2*x + 1) * n + 1][col] = 1  # C_{1, 2x+1}
            elif r == 1:
                W[(2*x + 0) * n + 1][col] = 1  # C_{1, 2x+0}
            elif r == 2:
                W[(2*x + 1) * n + 0][col] = 1  # C_{0, 2x+1}
            elif r == 3:
                W[(2*x + 0) * n + 0][col] = 1  # C_{0, 2x+0}
                W[(2*x + 0) * n + 1][col] = 1  # C_{1, 2x+0}
            elif r == 4:
                W[(2*x + 0) * n + 0][col] = -1 # C_{0, 2x+0}
                W[(2*x + 1) * n + 0][col] = 1  # C_{0, 2x+1}
            elif r == 5:
                W[(2*x + 1) * n + 1][col] = 1  # C_{1, 2x+1}
            elif r == 6:
                W[(2*x + 0) * n + 0][col] = 1  # C_{0, 2x+0}
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
