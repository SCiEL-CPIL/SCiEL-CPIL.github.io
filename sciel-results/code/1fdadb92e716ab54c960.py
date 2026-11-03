import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_val(mat, row, col, val):
        mat[row][col] += val

    # Term 1: M1 = (A11+A22)(B11+B22)
    # C11 += 1, C12 += -1, C21 += 1, C22 += -1
    set_val(U, 0, 0, 1); set_val(U, 4, 0, 1)
    set_val(V, 0, 0, 1); set_val(V, 6, 0, 1)
    set_val(W, 0, 0, 1); set_val(W, 1, 0, -1); set_val(W, 5, 0, 1); set_val(W, 6, 0, -1)
    
    # Term 2: M2 = (A21+A22)B11
    # C21 += -1, C22 += 1
    set_val(U, 3, 1, 1); set_val(U, 4, 1, 1)
    set_val(V, 0, 1, 1)
    set_val(W, 5, 1, -1); set_val(W, 6, 1, 1)
    
    # Term 3: M3 = A11(B12-B22)
    # C12 += 1, C22 += -1
    set_val(U, 0, 2, 1)
    set_val(V, 1, 2, 1); set_val(V, 6, 2, -1)
    set_val(W, 1, 2, 1); set_val(W, 6, 2, -1)
    
    # Term 4: M4 = A22(B21-B11)
    # C11 += 1, C21 += -1
    set_val(U, 4, 3, 1)
    set_val(V, 5, 3, 1); set_val(V, 0, 3, -1)
    set_val(W, 0, 3, 1); set_val(W, 5, 3, -1)
    
    # Term 5: M5 = (A11+A12)B22
    # C11 += -1, C12 += 1
    set_val(U, 0, 4, 1); set_val(U, 1, 4, 1)
    set_val(V, 6, 4, 1)
    set_val(W, 0, 4, -1); set_val(W, 1, 4, 1)
    
    # Term 6: M6 = (A21-A11)(B11+B12)
    # C12 += 1, C21 += 1
    set_val(U, 3, 5, 1); set_val(U, 0, 5, -1)
    set_val(V, 0, 5, 1); set_val(V, 1, 5, 1)
    set_val(W, 1, 5, 1); set_val(W, 5, 5, 1)
    
    # Term 7: M7 = (A12-A22)(B21+B22)
    # C11 += 1, C22 += 1
    set_val(U, 1, 6, 1); set_val(U, 4, 6, -1)
    set_val(V, 5, 6, 1); set_val(V, 6, 6, 1)
    set_val(W, 0, 6, 1); set_val(W, 6, 6, 1)
    
    # Terms 8-14: N1-N7 (Same as M1-M7 but B indices 2,3,7,8 and C indices 2,3,7,8)
    # N1
    set_val(U, 0, 7, 1); set_val(U, 4, 7, 1)
    set_val(V, 2, 7, 1); set_val(V, 7, 7, 1)
    set_val(W, 2, 7, 1); set_val(W, 3, 7, -1); set_val(W, 7, 7, 1); set_val(W, 8, 7, -1)
    
    # N2
    set_val(U, 3, 8, 1); set_val(U, 4, 8, 1)
    set_val(V, 2, 8, 1)
    set_val(W, 7, 8, -1); set_val(W, 8, 8, 1)
    
    # N3
    set_val(U, 0, 9, 1)
    set_val(V, 3, 9, 1); set_val(V, 7, 9, -1)
    set_val(W, 3, 9, 1); set_val(W, 8, 9, -1)
    
    # N4
    set_val(U, 4, 10, 1)
    set_val(V, 7, 10, 1); set_val(V, 2, 10, -1)
    set_val(W, 2, 10, 1); set_val(W, 7, 10, -1)
    
    # N5
    set_val(U, 0, 11, 1); set_val(U, 1, 11, 1)
    set_val(V, 7, 11, 1)
    set_val(W, 2, 11, -1); set_val(W, 3, 11, 1)
    
    # N6
    set_val(U, 3, 12, 1); set_val(U, 0, 12, -1)
    set_val(V, 2, 12, 1); set_val(V, 3, 12, 1)
    set_val(W, 3, 12, 1); set_val(W, 7, 12, 1)
    
    # N7
    set_val(U, 1, 13, 1); set_val(U, 4, 13, -1)
    set_val(V, 7, 13, 1); set_val(V, 7, 13, 1) # Wait, B23 is 7, B24 is 8.
    # N7 = (A12-A22)(B23+B24)
    # B23 index 7, B24 index 8.
    set_val(V, 7, 13, 1); set_val(V, 8, 13, 1)
    set_val(W, 2, 13, 1); set_val(W, 8, 13, 1)
    
    # Terms 15-18: P1-P4
    # P1 = A11 B15 -> C15
    set_val(U, 0, 14, 1)
    set_val(V, 4, 14, 1)
    set_val(W, 4, 14, 1)
    
    # P2 = A12 B25 -> C15
    set_val(U, 1, 15, 1)
    set_val(V, 9, 15, 1)
    set_val(W, 4, 15, 1)
    
    # P3 = A21 B15 -> C25
    set_val(U, 3, 16, 1)
    set_val(V, 4, 16, 1)
    set_val(W, 9, 16, 1)
    
    # P4 = A22 B25 -> C25
    set_val(U, 4, 17, 1)
    set_val(V, 9, 17, 1)
    set_val(W, 9, 17, 1)
    
    # Terms 19-28: Q11-Q25
    # Q11 = A13 B31 -> C11
    set_val(U, 2, 18, 1)
    set_val(V, 10, 18, 1)
    set_val(W, 0, 18, 1)
    
    # Q12 = A13 B32 -> C12
    set_val(U, 2, 19, 1)
    set_val(V, 11, 19, 1)
    set_val(W, 1, 19, 1)
    
    # Q13 = A13 B33 -> C13
    set_val(U, 2, 20, 1)
    set_val(V, 12, 20, 1)
    set_val(W, 2, 20, 1)
    
    # Q14 = A13 B34 -> C14
    set_val(U, 2, 21, 1)
    set_val(V, 13, 21, 1)
    set_val(W, 3, 21, 1)
    
    # Q15 = A13 B35 -> C15
    set_val(U, 2, 22, 1)
    set_val(V, 14, 22, 1)
    set_val(W, 4, 22, 1)
    
    # Q21 = A23 B31 -> C21
    set_val(U, 5, 23, 1)
    set_val(V, 10, 23, 1)
    set_val(W, 5, 23, 1)
    
    # Q22 = A23 B32 -> C22
    set_val(U, 5, 24, 1)
    set_val(V, 11, 24, 1)
    set_val(W, 6, 24, 1)
    
    # Q23 = A23 B33 -> C23
    set_val(U, 5, 25, 1)
    set_val(V, 12, 25, 1)
    set_val(W, 7, 25, 1)
    
    # Q24 = A23 B34 -> C24
    set_val(U, 5, 26, 1)
    set_val(V, 13, 26, 1)
    set_val(W, 8, 26, 1)
    
    # Q25 = A23 B35 -> C25
    set_val(U, 5, 27, 1)
    set_val(V, 14, 27, 1)
    set_val(W, 9, 27, 1)
    
    return {"U": U, "V": V, "W": W}
