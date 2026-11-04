import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_start, j_start, k_start):
        nonlocal col
        # M1 = (A11 + A22)(B21 + B22)
        U[i_start * m + j_start][col] += 1
        U[(i_start+1) * m + (j_start+1)][col] += 1
        V[(j_start+1) * p + k_start][col] += 1
        V[(j_start+1) * p + (k_start+1)][col] += 1
        W[k_start * n + i_start][col] += 1
        W[(k_start+1) * n + (i_start+1)][col] += 1
        col += 1
        
        # M2 = (A21 + A22) B11
        U[(i_start+1) * m + j_start][col] += 1
        U[(i_start+1) * m + (j_start+1)][col] += 1
        V[j_start * p + k_start][col] += 1
        W[(k_start+1) * n + i_start][col] += 1
        W[(k_start+1) * n + (i_start+1)][col] -= 1
        col += 1
        
        # M3 = A11 (B12 - B22)
        U[i_start * m + j_start][col] += 1
        V[j_start * p + (k_start+1)][col] += 1
        V[(j_start+1) * p + (k_start+1)][col] -= 1
        W[k_start * n + (i_start+1)][col] += 1
        W[(k_start+1) * n + (i_start+1)][col] += 1
        col += 1
        
        # M4 = A22 (B21 - B11)
        U[(i_start+1) * m + (j_start+1)][col] += 1
        V[(j_start+1) * p + k_start][col] += 1
        V[j_start * p + k_start][col] -= 1
        W[k_start * n + i_start][col] += 1
        W[(k_start+1) * n + i_start][col] += 1
        col += 1
        
        # M5 = (A11 + A12) B22
        U[i_start * m + j_start][col] += 1
        U[i_start * m + (j_start+1)][col] += 1
        V[(j_start+1) * p + (k_start+1)][col] += 1
        W[k_start * n + i_start][col] -= 1
        W[k_start * n + (i_start+1)][col] += 1
        col += 1
        
        # M6 = (A21 - A11) (B11 + B12)
        U[(i_start+1) * m + j_start][col] += 1
        U[i_start * m + j_start][col] -= 1
        V[j_start * p + k_start][col] += 1
        V[j_start * p + (k_start+1)][col] += 1
        W[(k_start+1) * n + (i_start+1)][col] += 1
        col += 1
        
        # M7 = (A12 - A22) (B21 + B22)
        U[i_start * m + (j_start+1)][col] += 1
        U[(i_start+1) * m + (j_start+1)][col] -= 1
        V[(j_start+1) * p + k_start][col] += 1
        V[(j_start+1) * p + (k_start+1)][col] += 1
        W[k_start * n + i_start][col] += 1
        col += 1

    # A2 B2: 4x2x6
    for i_s in [0, 2]:
        for k_s in [0, 2, 4]:
            add_strassen_block(i_s, 3, k_s)
            
    # A11 B11: 4x1x6
    for i in range(4):
        U[i * m + 0][42] += 1
    for k in range(6):
        V[0 * p + k][42] += 1
    for k in range(6):
        for i in range(4):
            W[k * n + i][42] += 1
    col += 1

    # A12 B12: 4x2x6
    for i_s in [0, 2]:
        for k_s in [0, 2, 4]:
            add_strassen_block(i_s, 1, k_s)

    return {"U": U, "V": V, "W": W}
