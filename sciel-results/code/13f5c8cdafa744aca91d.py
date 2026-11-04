import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4
    rank = 34
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        if coeff != 0:
            U[u_idx][col] += coeff
            V[v_idx][col] += coeff
            W[w_idx][col] += coeff
            col += 1
    
    # 1. A11 * B11 -> C11 (Strassen, Rank 7)
    # A11: 0,1,3,4. B11: 0,1,4,5. C11: 0,1,3,4.
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2: (A10+A11)B00 -> C10, C11(-)
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[3][col] = 1; W[4][col] = -1
    col += 1
    
    # M3: A00(B01-B11) -> C01, C11
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[1][col] = 1; W[4][col] = 1
    col += 1
    
    # M4: A11(B10-B00) -> C00, C10
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M5: (A00+A01)B11 -> C00(-), C01
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. A12 * B21 -> C11 (Rank 4)
    # A12: 2,5. B21: 8,9. C11: 0,1,3,4.
    add_term(2, 8, 0)
    add_term(2, 9, 1)
    add_term(5, 8, 3)
    add_term(5, 9, 4)
    
    # 3. A11 * B12 -> C12 (Strassen, Rank 7)
    # A11: 0,1,3,4. B12: 2,3,6,7. C12: 6,7,9,10.
    # M1: (A00+A11)(B00+B11) -> C00(6), C11(10)
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[10][col] = 1
    col += 1
    
    # M2: (A10+A11)B00 -> C10(9), C11(10)(-)
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[9][col] = 1; W[10][col] = -1
    col += 1
    
    # M3: A00(B01-B11) -> C01(7), C11(10)
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[7][col] = 1; W[10][col] = 1
    col += 1
    
    # M4: A11(B10-B00) -> C00(6), C10(9)
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[6][col] = 1; W[9][col] = 1
    col += 1
    
    # M5: (A00+A01)B11 -> C00(6)(-), C01(7)
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[6][col] = -1; W[7][col] = 1
    col += 1
    
    # M6: (A10-A00)(B00+B01) -> C11(10)
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[10][col] = 1
    col += 1
    
    # M7: (A01-A11)(B10+B11) -> C00(6)
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1
    col += 1
    
    # 4. A12 * B22 -> C12 (Rank 4)
    # A12: 2,5. B22: 10,11. C12: 6,7,9,10.
    add_term(2, 10, 6)
    add_term(2, 11, 7)
    add_term(5, 10, 9)
    add_term(5, 11, 10)
    
    # 5. A21 * B11 -> C21 (Rank 4)
    # A21: 6,7. B11: 0,1,4,5. C21: 2,5.
    add_term(6, 0, 2)
    add_term(6, 1, 5)
    add_term(7, 4, 2)
    add_term(7, 5, 5)
    
    # 6. A22 * B21 -> C21 (Rank 2)
    # A22: 8. B21: 8,9. C21: 2,5.
    add_term(8, 8, 2)
    add_term(8, 9, 5)
    
    # 7. A21 * B12 -> C22 (Rank 4)
    # A21: 6,7. B12: 2,3,6,7. C22: 8,11.
    add_term(6, 2, 8)
    add_term(6, 3, 11)
    add_term(7, 6, 8)
    add_term(7, 7, 11)
    
    # 8. A22 * B22 -> C22 (Rank 2)
    # A22: 8. B22: 10,11. C22: 8,11.
    add_term(8, 10, 8)
    add_term(8, 11, 11)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
