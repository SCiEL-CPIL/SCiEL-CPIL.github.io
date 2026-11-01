"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit
    
    n, m, p = 2, 2, 13
    
    # Strategy: 6 blocks of 2 columns (12 cols) + 1 block of 1 column
    # For each 2-column block, use rank-7 (Strassen) decomposition
    # For the 1-column block, use rank-4 (schoolbook)
    # Total: 6*7 + 4 = 46
    
    # Strassen's algorithm for 2x2 * 2x2:
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a10+a11)*b00
    # P3 = a00*(b10-b11)
    # P4 = a11*(b01-b00)
    # P5 = (a00+a01)*b11
    # P6 = (a10-a00)*(b00+b10)
    # P7 = (a01-a11)*(b01+b11)
    # C00 = P1 - P4 + P6 - P7
    # C01 = P3 + P1 - P2 + P6
    # C10 = P2 + P4 - P3 + P5
    # C11 = P1 - P4 + P2 - P6
    
    # Define the 7 Strassen products as (L_A, L_B, W) triples
    # L_A: coefficients for [a00, a01, a10, a11]
    # L_B: coefficients for [b00, b01, b10, b11]
    # W: coefficients for [C00, C01, C10, C11]
    strassen_products = [
        ([1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, -1, 1, 1]),  # P2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 1, 0]),  # P3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, -1]), # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]),  # P5
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, -1]), # P6
        ([0, 1, 0, -1], [0, 0, 0, 1], [1, 0, 0, 0]),  # P7
    ]
    
    # Build the full decomposition
    rank = 6 * 7 + 4  # 46
    U_full = [[0] * rank for _ in range(n * m)]      # 4 x 46
    V_full = [[0] * rank for _ in range(m * p)]      # 26 x 46
    W_full = [[0] * rank for _ in range(p * n)]      # 26 x 46
    
    col = 0
    # 6 blocks of 2 columns
    for block in range(6):
        j0 = 2 * block
        j1 = 2 * block + 1
        for t in range(7):
            LA, LB, WC = strassen_products[t]
            # U: index = i*2+k, so [a00, a01, a10, a11]
            U_full[0][col] = LA[0]  # a00
            U_full[1][col] = LA[1]  # a01
            U_full[2][col] = LA[2]  # a10
            U_full[3][col] = LA[3]  # a11
            # V: index = k*13+j
            # b00 -> k=0,j=j0; b01 -> k=0,j=j1; b10 -> k=1,j=j0; b11 -> k=1,j=j1
            V_full[0 * 13 + j0][col] = LB[0]  # b00
            V_full[0 * 13 + j1][col] = LB[1]  # b01
            V_full[1 * 13 + j0][col] = LB[2]  # b10
            V_full[1 * 13 + j1][col] = LB[3]  # b11
            # W: index = j*2+i
            # C00 -> j=j0,i=0; C01 -> j=j0,i=1; C10 -> j=j1,i=0; C11 -> j=j1,i=1
            W_full[j0 * 2 + 0][col] = WC[0]  # C00
            W_full[j0 * 2 + 1][col] = WC[1]  # C01
            W_full[j1 * 2 + 0][col] = WC[2]  # C10
            W_full[j1 * 2 + 1][col] = WC[3]  # C11
            col += 1
    
    # 1 block of 1 column (j=12)
    j = 12
    # Schoolbook for 2x2 * 2x1: 4 products
    # C[j,0] = a00*b[j,0] + a01*b[j,1]
    # C[j,1] = a10*b[j,0] + a11*b[j,1]
    # Product 1: a00 * b[j,0] -> C[j,0] += 1
    U_full[0][col] = 1
    V_full[0 * 13 + j][col] = 1
    W_full[j * 2 + 0][col] = 1
    col += 1
    # Product 2: a01 * b[j,1] -> C[j,0] += 1
    U_full[1][col] = 1
    V_full[1 * 13 + j][col] = 1
    W_full[j * 2 + 0][col] = 1
    col += 1
    # Product 3: a10 * b[j,0] -> C[j,1] += 1
    U_full[2][col] = 1
    V_full[0 * 13 + j][col] = 1
    W_full[j * 2 + 1][col] = 1
    col += 1
    # Product 4: a11 * b[j,1] -> C[j,1] += 1
    U_full[3][col] = 1
    V_full[1 * 13 + j][col] = 1
    W_full[j * 2 + 1][col] = 1
    col += 1
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
