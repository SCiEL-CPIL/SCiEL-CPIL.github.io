import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Block decomposition 3+2
    # A is 5x5, split into A00(3x3), A01(3x2), A10(2x3), A11(2x2)
    # B is 5x5, split into B00(3x3), B01(3x2), B10(2x3), B11(2x2)
    # C is 5x5, split into C00(3x3), C01(3x2), C10(2x3), C11(2x2)
    
    # Indices for A (row-major i*5+j)
    # A00: i in 0..2, j in 0..2
    A00 = [i*5+j for i in range(3) for j in range(3)]
    # A01: i in 0..2, j in 3..4
    A01 = [i*5+j for i in range(3) for j in range(3, 5)]
    # A10: i in 3..4, j in 0..2
    A10 = [i*5+j for i in range(3, 5) for j in range(3)]
    # A11: i in 3..4, j in 3..4
    A11 = [i*5+j for i in range(3, 5) for j in range(3, 5)]
    
    # Indices for B (row-major j*5+k)
    # B00: j in 0..2, k in 0..2
    B00 = [j*5+k for j in range(3) for k in range(3)]
    # B01: j in 0..2, k in 3..4
    B01 = [j*5+k for j in range(3) for k in range(3, 5)]
    # B10: j in 3..4, k in 0..2
    B10 = [j*5+k for j in range(3, 5) for k in range(3)]
    # B11: j in 3..4, k in 3..4
    B11 = [j*5+k for j in range(3, 5) for k in range(3, 5)]
    
    # Indices for C (row-major k*5+i)
    # C00: k in 0..2, i in 0..2
    C00 = [k*5+i for k in range(3) for i in range(3)]
    # C01: k in 0..2, i in 3..4
    C01 = [k*5+i for k in range(3) for i in range(3, 5)]
    # C10: k in 3..4, i in 0..2
    C10 = [k*5+i for k in range(3, 5) for i in range(3)]
    # C11: k in 3..4, i in 3..4
    C11 = [k*5+i for k in range(3, 5) for i in range(3, 5)]

    # Calculate total rank
    rank_A00_B00 = 3*3*3 # 27
    rank_A01_B10 = 3*2*3 # 18
    rank_A01_B11 = 3*2*2 # 12
    rank_A10_B00 = 2*3*3 # 18
    rank_A10_B01 = 2*3*2 # 12
    rank_A11_B10 = 2*2*3 # 12
    rank_A11_B11 = 7     # Strassen
    
    total_rank = rank_A00_B00 + rank_A01_B10 + rank_A01_B11 + rank_A10_B00 + rank_A10_B01 + rank_A11_B10 + rank_A11_B11
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolblock(A_idx, B_idx, C_idx, m_size, n_size, p_size):
        nonlocal col
        for i in range(m_size):
            for j in range(n_size):
                for k in range(p_size):
                    a_idx = A_idx[i*n_size + j]
                    b_idx = B_idx[j*p_size + k]
                    c_idx = C_idx[k*m_size + i]
                    U[a_idx][col] = 1
                    V[b_idx][col] = 1
                    W[c_idx][col] = 1
                    col += 1

    def add_strassen_block(A_idx, B_idx, C_idx):
        # A_idx, B_idx, C_idx are lists of 4 indices for 2x2 blocks
        # A00, A01, A10, A11
        # B00, B01, B10, B11
        # C00, C01, C10, C11
        nonlocal col
        
        A00, A01, A10, A11 = A_idx
        B00, B01, B10, B11 = B_idx
        C00, C01, C10, C11 = C_idx
        
        # M1 = (A00+A11)(B00+B11)
        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C11][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = -1
        col += 1
        
        # M3 = A00(B10-B11)
        U[A00][col] = 1
        V[B10][col] = 1; V[B11][col] = -1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1
        
        # M4 = A11(B01-B00)
        U[A11][col] = 1
        V[B01][col] = 1; V[B00][col] = -1
        W[C00][col] = 1; W[C10][col] = 1
        col += 1
        
        # M5 = (A00+A10)B11
        U[A00][col] = 1; U[A10][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1; W[C01][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B10)
        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B10][col] = 1
        W[C11][col] = 1
        col += 1
        
        # M7 = (A11-A01)(B01+B11)
        U[A11][col] = 1; U[A01][col] = -1
        V[B01][col] = 1; V[B11][col] = 1
        W[C00][col] = 1
        col += 1

    # Add blocks
    # 1. A00 B00 -> C00
    add_schoolblock(A00, B00, C00, 3, 3, 3)
    
    # 2. A01 B10 -> C00
    add_schoolblock(A01, B10, C00, 3, 2, 3)
    
    # 3. A01 B11 -> C01
    add_schoolblock(A01, B11, C01, 3, 2, 2)
    
    # 4. A10 B00 -> C10
    add_schoolblock(A10, B00, C10, 2, 3, 3)
    
    # 5. A10 B01 -> C11
    add_schoolblock(A10, B01, C11, 2, 3, 2)
    
    # 6. A11 B10 -> C10
    add_schoolblock(A11, B10, C10, 2, 2, 3)
    
    # 7. A11 B11 -> C11 (Strassen)
    # We need to map the 2x2 block indices to the Strassen helper
    # A11 is 2x2: indices 18,19,23,24
    # A11_00=18, A11_01=19, A11_10=23, A11_11=24
    # B11 is 2x2: indices 18,19,23,24
    # B11_00=18, B11_01=19, B11_10=23, B11_11=24
    # C11 is 2x2: indices 18,19,23,24
    # C11_00=18, C11_01=19, C11_10=23, C11_11=24
    
    A11_sub = [18, 19, 23, 24]
    B11_sub = [18, 19, 23, 24]
    C11_sub = [18, 19, 23, 24]
    
    add_strassen_block(A11_sub, B11_sub, C11_sub)

    return {"U": U, "V": V, "W": W}
