import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(j_A_start, j_B_start, k_start, col_start):
        # M1
        U[0 * m + j_A_start][col_start] += 1
        V[j_B_start * p + k_start][col_start] += 1
        V[(j_B_start + 1) * p + (k_start + 1)][col_start] += 1
        W[k_start * n + 0][col_start] += 1
        W[(k_start + 1) * n + 1][col_start] += 1
        
        # M2
        U[0 * m + j_A_start][col_start + 1] += 1
        U[1 * m + j_A_start][col_start + 1] += 1
        V[(j_B_start + 1) * p + (k_start + 1)][col_start + 1] += 1
        W[k_start * n + 1][col_start + 1] += 1
        W[(k_start + 1) * n + 1][col_start + 1] -= 1
        
        # M3
        U[0 * m + (j_A_start + 1)][col_start + 2] += 1
        U[1 * m + (j_A_start + 1)][col_start + 2] += 1
        V[j_B_start * p + k_start][col_start + 2] += 1
        W[(k_start + 1) * n + 0][col_start + 2] += 1
        W[(k_start + 1) * n + 1][col_start + 2] += 1
        
        # M4
        U[1 * m + (j_A_start + 1)][col_start + 3] += 1
        V[j_B_start * p + (k_start + 1)][col_start + 3] += 1
        V[(j_B_start + 1) * p + k_start][col_start + 3] += 1
        W[k_start * n + 0][col_start + 3] -= 1
        W[(k_start + 1) * n + 0][col_start + 3] += 1
        W[k_start * n + 1][col_start + 3] += 1
        W[(k_start + 1) * n + 1][col_start + 3] += 1
        
        # M5
        U[0 * m + j_A_start][col_start + 4] += 1
        U[0 * m + (j_A_start + 1)][col_start + 4] += 1
        V[(j_B_start + 1) * p + k_start][col_start + 4] += 1
        W[k_start * n + 0][col_start + 4] += 1
        W[(k_start + 1) * n + 0][col_start + 4] -= 1
        
        # M6
        U[1 * m + j_A_start][col_start + 5] += 1
        U[0 * m + j_A_start][col_start + 5] -= 1
        V[j_B_start * p + k_start][col_start + 5] += 1
        V[j_B_start * p + (k_start + 1)][col_start + 5] += 1
        W[(k_start + 1) * n + 0][col_start + 5] -= 1
        W[k_start * n + 1][col_start + 5] += 1
        W[(k_start + 1) * n + 1][col_start + 5] += 1
        
        # M7
        U[0 * m + (j_A_start + 1)][col_start + 6] += 1
        U[1 * m + (j_A_start + 1)][col_start + 6] -= 1
        V[(j_B_start + 1) * p + k_start][col_start + 6] += 1
        V[(j_B_start + 1) * p + (k_start + 1)][col_start + 6] += 1
        W[k_start * n + 0][col_start + 6] += 1
        
        return col_start + 7

    col = 0
    col = add_strassen_block(0, 0, 0, col)
    col = add_strassen_block(2, 2, 0, col)
    col = add_strassen_block(0, 0, 2, col)
    col = add_strassen_block(2, 2, 2, col)
    col = add_strassen_block(0, 0, 4, col)
    col = add_strassen_block(2, 2, 4, col)
    
    return {"U": U, "V": V, "W": W}
