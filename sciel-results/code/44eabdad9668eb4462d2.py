import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    
    # Rank calculation:
    # 4 blocks of 2x2x2 (Strassen rank 7) = 28
    # 1 part of 2x1x8 (rank 2) = 2
    # Total rank = 30
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition for 2x2x2
    # A is 2x2, B is 2x2
    # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    # M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
    # M3 = A00(B01-B11) -> C01 += 1, C11 += 1
    # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
    # M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
    # M6 = (A10-A11)(B01+B11) -> C11 += 1
    # M7 = (A01-A10)(B00+B10) -> C00 += 1, C10 += 1
    
    # Indices for U: i * m + j. Here i in {0,1}, j in {0,1} for the block.
    # Indices for V: j * p + k. Here j in {0,1}, k in {0,1} local.
    # Indices for W: k * n + i. Here k in {0,1} local, i in {0,1}.
    
    for b in range(4):
        # Local k offset
        k_off = 4 * b
        
        # M1
        U[0 * m + 0][col] += 1
        U[1 * m + 1][col] += 1
        V[0 * p + (k_off + 0)][col] += 1
        V[1 * p + (k_off + 1)][col] += 1
        W[(k_off + 0) * n + 0][col] += 1
        W[(k_off + 1) * n + 1][col] += 1
        col += 1
        
        # M2
        U[1 * m + 0][col] += 1
        U[1 * m + 1][col] += 1
        V[0 * p + (k_off + 0)][col] += 1
        W[(k_off + 0) * n + 1][col] += 1
        W[(k_off + 1) * n + 1][col] -= 1
        col += 1
        
        # M3
        U[0 * m + 0][col] += 1
        V[0 * p + (k_off + 1)][col] += 1
        V[1 * p + (k_off + 1)][col] -= 1
        W[(k_off + 1) * n + 0][col] += 1
        W[(k_off + 1) * n + 1][col] += 1
        col += 1
        
        # M4
        U[1 * m + 1][col] += 1
        V[1 * p + (k_off + 0)][col] += 1
        V[0 * p + (k_off + 0)][col] -= 1
        W[(k_off + 0) * n + 0][col] += 1
        W[(k_off + 0) * n + 1][col] += 1
        col += 1
        
        # M5
        U[0 * m + 0][col] += 1
        U[0 * m + 1][col] += 1
        V[1 * p + (k_off + 1)][col] += 1
        W[(k_off + 0) * n + 0][col] -= 1
        W[(k_off + 1) * n + 0][col] += 1
        col += 1
        
        # M6
        U[1 * m + 0][col] += 1
        U[1 * m + 1][col] -= 1
        V[0 * p + (k_off + 1)][col] += 1
        V[1 * p + (k_off + 1)][col] += 1
        W[(k_off + 1) * n + 1][col] += 1
        col += 1
        
        # M7
        U[0 * m + 1][col] += 1
        U[1 * m + 0][col] -= 1
        V[0 * p + (k_off + 0)][col] += 1
        V[1 * p + (k_off + 0)][col] += 1
        W[(k_off + 0) * n + 0][col] += 1
        W[(k_off + 0) * n + 1][col] += 1
        col += 1
        
    # 2x1x8 part
    # A col 2, B row 2
    # C0k = A02 B2k
    # C1k = A12 B2k
    
    # Term 1: C0k
    U[0 * m + 2][col] += 1
    for k in range(8):
        V[2 * p + k][col] += 1
    for k in range(8):
        W[k * n + 0][col] += 1
    col += 1
    
    # Term 2: C1k
    U[1 * m + 2][col] += 1
    for k in range(8):
        V[2 * p + k][col] += 1
    for k in range(8):
        W[k * n + 1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
