"""
Rank-optimized tensor decomposition for 5x5x5 matrix multiplication.
Uses block partitioning with Strassen (rank 7) for 2x2 and schoolbook for 3x3 and cross terms.
"""
import time

_START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def _time_remaining():
    return DEADLINE - SAFETY_MARGIN - (time.time() - _START_TIME)

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    
    u_rows = []
    v_rows = []
    w_rows = []
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        nonlocal col
        if coeff != 0.0:
            u_rows.append((u_idx, col, coeff))
            v_rows.append((v_idx, col, coeff))
            w_rows.append((w_idx, col, coeff))
            col += 1
    
    # ===== 3x3 block: A[0:3,0:3] * B[0:3,0:3] -> C[0:3,0:3] (schoolbook, rank 27) =====
    for i in range(3):
        for j in range(3):
            for k in range(3):
                add_term(i * 5 + k, k * 5 + j, j * 5 + i)
    
    # ===== 2x2 block: Strassen (rank 7) =====
    # Local (i,j) -> global (3+i, 3+j), so global row = (3+i)*5 + (3+j)
    def g2(i, j):
        return (3 + i) * 5 + (3 + j)
    
    # M1 = (A00+A11)*(B00+B11) -> contributes +1 to C00, +1 to C11
    # U: g2(0,0)=1, g2(1,1)=1
    # V: g2(0,0)=1, g2(1,1)=1  
    # W: g2(0,0)=+1, g2(1,1)=+1
    u_rows.append((g2(0,0), col, 1.0))
    u_rows.append((g2(1,1), col, 1.0))
    v_rows.append((g2(0,0), col, 1.0))
    v_rows.append((g2(1,1), col, 1.0))
    w_rows.append((g2(0,0), col, 1.0))
    w_rows.append((g2(1,1), col, 1.0))
    col += 1
    
    # M2 = (A10+A11)*B00 -> contributes +1 to C10, -1 to C11
    u_rows.append((g2(1,0), col, 1.0))
    u_rows.append((g2(1,1), col, 1.0))
    v_rows.append((g2(0,0), col, 1.0))
    w_rows.append((g2(1,0), col, 1.0))
    w_rows.append((g2(1,1), col, -1.0))
    col += 1
    
    # M3 = A00*(B01-B11) -> contributes +1 to C01, +1 to C11
    u_rows.append((g2(0,0), col, 1.0))
    v_rows.append((g2(0,1), col, 1.0))
    v_rows.append((g2(1,1), col, -1.0))
    w_rows.append((g2(0,1), col, 1.0))
    w_rows.append((g2(1,1), col, 1.0))
    col += 1
    
    # M4 = A11*(B10-B00) -> contributes +1 to C10, +1 to C00
    u_rows.append((g2(1,1), col, 1.0))
    v_rows.append((g2(1,0), col, 1.0))
    v_rows.append((g2(0,0), col, -1.0))
    w_rows.append((g2(1,0), col, 1.0))
    w_rows.append((g2(0,0), col, 1.0))
    col += 1
    
    # M5 = (A00+A01)*B11 -> contributes -1 to C00, +1 to C01
    u_rows.append((g2(0,0), col, 1.0))
    u_rows.append((g2(0,1), col, 1.0))
    v_rows.append((g2(1,1), col, 1.0))
    w_rows.append((g2(0,0), col, -1.0))
    w_rows.append((g2(0,1), col, 1.0))
    col += 1
    
    # M6 = (A10-A00)*(B00+B01) -> contributes +1 to C11
    u_rows.append((g2(1,0), col, 1.0))
    u_rows.append((g2(0,0), col, -1.0))
    v_rows.append((g2(0,0), col, 1.0))
    v_rows.append((g2(0,1), col, 1.0))
    w_rows.append((g2(1,1), col, 1.0))
    col += 1
    
    # M7 = (A01-A11)*(B10+B11) -> contributes +1 to C00, +1 to C11
    u_rows.append((g2(0,1), col, 1.0))
    u_rows.append((g2(1,1), col, -1.0))
    v_rows.append((g2(1,0), col, 1.0))
    v_rows.append((g2(1,1), col, 1.0))
    w_rows.append((g2(0,0), col, 1.0))
    w_rows.append((g2(1,1), col, 1.0))
    col += 1
    
    # ===== Cross: A[0:3,3:5] * B[3:5,0:3] -> C[0:3,0:3] (3x2 * 2x3, rank 18) =====
    for i in range(3):
        for j in range(3):
            for k in range(2):
                add_term(i * 5 + (3 + k), (3 + k) * 5 + j, j * 5 + i)
    
    # ===== Cross: A[0:3,0:3] * B[0:3,3:5] -> C[0:3,3:5] (3x3 * 3x2, rank 18) =====
    for i in range(3):
        for j in range(2):
            for k in range(3):
                add_term(i * 5 + k, k * 5 + (3 + j), (3 + j) * 5 + i)
    
    # ===== Cross: A[3:5,0:3] * B[0:3,0:3] -> C[3:5,0:3] (2x3 * 3x3, rank 18) =====
    for i in range(2):
        for j in range(3):
            for k in range(3):
                add_term((3 + i) * 5 + k, k * 5 + j, j * 5 + (3 + i))
    
    # ===== Cross: A[0:3,3:5] * B[3:5,3:5] -> C[0:3,3:5] (3x2 * 2x2, rank 12) =====
    for i in range(3):
        for j in range(2):
            for k in range(2):
                add_term(i * 5 + (3 + k), (3 + k) * 5 + (3 + j), (3 + j) * 5 + i)
    
    # ===== Cross: A[3:5,0:3] * B[0:3,3:5] -> C[3:5,3:5] (2x3 * 3x2, rank 12) =====
    for i in range(2):
        for j in range(2):
            for k in range(3):
                add_term((3 + i) * 5 + k, k * 5 + (3 + j), (3 + j) * 5 + (3 + i))
    
    # ===== Cross: A[3:5,3:5] * B[3:5,0:3] -> C[3:5,0:3] (2x2 * 2x3, rank 12) =====
    for i in range(2):
        for j in range(3):
            for k in range(2):
                add_term((3 + i) * 5 + (3 + k), (3 + k) * 5 + j, j * 5 + (3 + i))
    
    actual_rank = col
    
    U = [[0.0] * actual_rank for _ in range(25)]
    V = [[0.0] * actual_rank for _ in range(25)]
    W = [[0.0] * actual_rank for _ in range(25)]
    
    for (row, c, val) in u_rows:
        U[row][c] = val
    for (row, c, val) in v_rows:
        V[row][c] = val
    for (row, c, val) in w_rows:
        W[row][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
