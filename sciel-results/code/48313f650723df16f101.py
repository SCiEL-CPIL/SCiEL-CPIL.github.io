import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    # n=2, m=3, p=5
    # U: 6 x R
    # V: 15 x R
    # W: 10 x R
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for 2x2 * 2x2
    # a_i, a_j: indices for A (i in 0..1, j in j_offset..j_offset+1)
    # b_i, b_j: indices for B (j in j_offset..j_offset+1, k in k_offset..k_offset+1)
    # c_i, c_k: indices for C (i in 0..1, k in k_offset..k_offset+1)
    def add_strassen_block(a_j_offset, b_j_offset, b_k_offset):
        nonlocal col
        # A indices: (0, a_j_offset), (0, a_j_offset+1), (1, a_j_offset), (1, a_j_offset+1)
        # B indices: (b_j_offset, b_k_offset), (b_j_offset, b_k_offset+1), (b_j_offset+1, b_k_offset), (b_j_offset+1, b_k_offset+1)
        # C indices: (0, b_k_offset), (0, b_k_offset+1), (1, b_k_offset), (1, b_k_offset+1)
        
        A00 = a_j_offset
        A01 = a_j_offset + 1
        A10 = 3 + a_j_offset # i=1, j=0 -> 1*3+0=3
        A11 = 3 + a_j_offset + 1
        
        B00 = b_j_offset * p + b_k_offset
        B01 = b_j_offset * p + b_k_offset + 1
        B10 = (b_j_offset + 1) * p + b_k_offset
        B11 = (b_j_offset + 1) * p + b_k_offset + 1
        
        C00 = b_k_offset * n + 0
        C01 = b_k_offset * n + 1
        C10 = b_k_offset * n + 2
        C11 = b_k_offset * n + 3
        
        # M1 = (A00+A11)(B00+B11)
        U[A00][col] = 1
        U[A11][col] = 1
        V[B00][col] = 1
        V[B11][col] = 1
        W[C00][col] = 1
        W[C11][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[A10][col] = 1
        U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1
        W[C11][col] = -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[A00][col] = 1
        V[B01][col] = 1
        V[B11][col] = -1
        W[C01][col] = 1
        W[C11][col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[A11][col] = 1
        V[B10][col] = 1
        V[B00][col] = -1
        W[C00][col] = 1
        W[C10][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[A00][col] = 1
        U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1
        W[C01][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[A10][col] = 1
        U[A00][col] = -1
        V[B00][col] = 1
        V[B01][col] = 1
        W[C11][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[A01][col] = 1
        U[A11][col] = -1
        V[B10][col] = 1
        V[B11][col] = 1
        W[C00][col] = 1
        W[C10][col] = -1
        col += 1
        
    # Block 1: A_L (j=0,1) * B_L1 (j=0,1, k=0,1)
    # a_j_offset = 0, b_j_offset = 0, b_k_offset = 0
    add_strassen_block(0, 0, 0)
    
    # Block 2: A_L (j=0,1) * B_L2 (j=0,1, k=2,3)
    # a_j_offset = 0, b_j_offset = 0, b_k_offset = 2
    add_strassen_block(0, 0, 2)
    
    # Block 3: A_L (j=0,1) * B_L3 (j=0,1, k=4)
    # Rank 4
    # M1 = A00 B04
    U[0][col] = 1
    V[0*5+4][col] = 1
    W[4*2+0][col] = 1
    col += 1
    
    # M2 = A01 B14
    U[1][col] = 1
    V[1*5+4][col] = 1
    W[4*2+0][col] = 1
    col += 1
    
    # M3 = A10 B04
    U[3][col] = 1
    V[0*5+4][col] = 1
    W[4*2+1][col] = 1
    col += 1
    
    # M4 = A11 B14
    U[4][col] = 1
    V[1*5+4][col] = 1
    W[4*2+1][col] = 1
    col += 1
    
    # Block 4: A_R (j=2) * B_R (j=2, k=0..4)
    # Rank 10
    for k in range(5):
        # M = A02 B2k
        U[2][col] = 1 # i=0, j=2 -> 0*3+2=2
        V[2*5+k][col] = 1 # j=2, k=k -> 2*5+k
        W[k*2+0][col] = 1 # k=k, i=0 -> k*2+0
        col += 1
        
        # M = A12 B2k
        U[5][col] = 1 # i=1, j=2 -> 1*3+2=5
        V[2*5+k][col] = 1
        W[k*2+1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
