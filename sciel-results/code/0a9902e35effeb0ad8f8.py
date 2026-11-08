"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    
    # Rank calculation
    # A1 (2x5) * B (5x6)
    # Split B into 3 blocks of 5x2
    # Each 2x5x2 block:
    #   Split 5 into 2+2+1
    #   2x2x2 (rank 7) + 2x2x2 (rank 7) + 2x1x2 (rank 4) = 18
    # 3 blocks * 18 = 54
    # A2 (1x5) * B (5x6)
    # Split B into 3 blocks of 5x2
    # Each 1x5x2 block:
    #   Split 5 into 2+2+1
    #   1x2x2 (rank 4) + 1x2x2 (rank 4) + 1x1x2 (rank 2) = 10
    # 3 blocks * 10 = 30
    # Total rank 84
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add 2x2x2 Strassen block
    def add_strassen_2x2(i_start, j_start, k_start, c):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[i_start * m + j_start][c] += 1
        U[(i_start + 1) * m + (j_start + 1)][c] += 1
        V[j_start * p + k_start][c] += 1
        V[(j_start + 1) * p + (k_start + 1)][c] += 1
        W[k_start * n + i_start][c] += 1
        W[(k_start + 1) * n + (i_start + 1)][c] += 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[(i_start + 1) * m + j_start][c] += 1
        U[(i_start + 1) * m + (j_start + 1)][c] += 1
        V[j_start * p + k_start][c] += 1
        W[k_start * n + (i_start + 1)][c] += 1
        col += 1
        
        # M3 = A00(B01 - B11)
        U[i_start * m + j_start][c] += 1
        V[j_start * p + (k_start + 1)][c] += 1
        V[(j_start + 1) * p + (k_start + 1)][c] -= 1
        W[(k_start + 1) * n + i_start][c] += 1
        col += 1
        
        # M4 = A11(B10 - B00)
        U[(i_start + 1) * m + (j_start + 1)][c] += 1
        V[(j_start + 1) * p + k_start][c] += 1
        V[j_start * p + k_start][c] -= 1
        W[k_start * n + (i_start + 1)][c] += 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[i_start * m + j_start][c] += 1
        U[i_start * m + (j_start + 1)][c] += 1
        V[(j_start + 1) * p + (k_start + 1)][c] += 1
        W[(k_start + 1) * n + i_start][c] += 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[(i_start + 1) * m + j_start][c] += 1
        U[i_start * m + j_start][c] -= 1
        V[j_start * p + k_start][c] += 1
        V[j_start * p + (k_start + 1)][c] += 1
        W[k_start * n + (i_start + 1)][c] += 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[i_start * m + (j_start + 1)][c] += 1
        U[(i_start + 1) * m + (j_start + 1)][c] -= 1
        V[(j_start + 1) * p + k_start][c] += 1
        V[(j_start + 1) * p + (k_start + 1)][c] += 1
        W[(k_start + 1) * n + i_start][c] += 1
        col += 1

    # Helper to add 2x1x2 block (rank 4)
    def add_2x1x2(i_start, j_start, k_start, c):
        nonlocal col
        # M1 = A00 B00
        U[i_start * m + j_start][c] += 1
        V[j_start * p + k_start][c] += 1
        W[k_start * n + i_start][c] += 1
        col += 1
        
        # M2 = A00 B01
        U[i_start * m + j_start][c] += 1
        V[j_start * p + (k_start + 1)][c] += 1
        W[(k_start + 1) * n + i_start][c] += 1
        col += 1
        
        # M3 = A10 B00
        U[(i_start + 1) * m + j_start][c] += 1
        V[j_start * p + k_start][c] += 1
        W[k_start * n + (i_start + 1)][c] += 1
        col += 1
        
        # M4 = A10 B01
        U[(i_start + 1) * m + j_start][c] += 1
        V[j_start * p + (k_start + 1)][c] += 1
        W[(k_start + 1) * n + (i_start + 1)][c] += 1
        col += 1

    # Helper to add 1x2x2 block (rank 4)
    def add_1x2x2(i_start, j_start, k_start, c):
        nonlocal col
        # M1 = A00 B00
        U[i_start * m + j_start][c] += 1
        V[j_start * p + k_start][c] += 1
        W[k_start * n + i_start][c] += 1
        col += 1
        
        # M2 = A00 B01
        U[i_start * m + j_start][c] += 1
        V[j_start * p + (k_start + 1)][c] += 1
        W[(k_start + 1) * n + i_start][c] += 1
        col += 1
        
        # M3 = A01 B10
        U[i_start * m + (j_start + 1)][c] += 1
        V[(j_start + 1) * p + k_start][c] += 1
        W[k_start * n + i_start][c] += 1
        col += 1
        
        # M4 = A01 B11
        U[i_start * m + (j_start + 1)][c] += 1
        V[(j_start + 1) * p + (k_start + 1)][c] += 1
        W[(k_start + 1) * n + i_start][c] += 1
        col += 1

    # Helper to add 1x1x2 block (rank 2)
    def add_1x1x2(i_start, j_start, k_start, c):
        nonlocal col
        # M1 = A00 B00
        U[i_start * m + j_start][c] += 1
        V[j_start * p + k_start][c] += 1
        W[k_start * n + i_start][c] += 1
        col += 1
        
        # M2 = A00 B01
        U[i_start * m + j_start][c] += 1
        V[j_start * p + (k_start + 1)][c] += 1
        W[(k_start + 1) * n + i_start][c] += 1
        col += 1

    # Process A1 (rows 0, 1)
    for k_block in range(3):
        k_start = k_block * 2
        if time.time() > deadline: break
        add_strassen_2x2(0, 0, k_start, col)
        add_strassen_2x2(0, 2, k_start, col)
        add_2x1x2(0, 4, k_start, col)

    # Process A2 (row 2)
    for k_block in range(3):
        k_start = k_block * 2
        if time.time() > deadline: break
        add_1x2x2(2, 0, k_start, col)
        add_1x2x2(2, 2, k_start, col)
        add_1x1x2(2, 4, k_start, col)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
