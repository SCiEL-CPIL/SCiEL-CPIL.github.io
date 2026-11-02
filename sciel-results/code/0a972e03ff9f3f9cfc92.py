import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Indexing:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition for 2x2x2
    # Local indices 0,1
    # U_local: 4 entries (0,0), (0,1), (1,0), (1,1)
    # V_local: 4 entries
    # W_local: 4 entries
    
    # Terms: (U_vec, V_vec, W_vec)
    # Flattened row-major
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [1, -1, 0, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([0, -1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    def add_strassen_block(rA, cA, rB, cB):
        nonlocal col
        for u_loc, v_loc, w_loc in strassen_terms:
            for i in range(2):
                for j in range(2):
                    val = u_loc[i * 2 + j]
                    if val != 0:
                        U[rA + i][cA + j] += val # Wait, U is [i*m+j]
                        # Correct index: U[(rA+i)*m + (cA+j)]
            
            # Correction: U index is global i, global j
            # U[global_i * m + global_j]
            
            # Let's rewrite the loop properly
            for i in range(2):
                for j in range(2):
                    val = u_loc[i * 2 + j]
                    if val != 0:
                        U[(rA + i) * m + (cA + j)][col] = val
            
            for j in range(2):
                for k in range(2):
                    val = v_loc[j * 2 + k]
                    if val != 0:
                        V[(rB + j) * p + (cB + k)][col] = val
            
            for k in range(2):
                for i in range(2):
                    val = w_loc[k * 2 + i]
                    if val != 0:
                        W[(rB + k) * n + (rA + i)][col] = val
            
            col += 1

    def add_schoolbook_223_block(rA, cA, rB, cB):
        # A is 2x2, B is 2x3
        # i in 0..1, j in 0..1, k in 0..2
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[(rA + i) * m + (cA + j)][col] = 1
                    V[(rB + j) * p + (cB + k)][col] = 1
                    W[(rB + k) * n + (rA + i)][col] = 1
                    col += 1

    # 4 Strassen blocks for C11 and C21 (2x2 outputs)
    # C11 = A00 B00 + A01 B10
    # C21 = A10 B00 + A11 B10
    
    # A00: rA=0, cA=0. B00: rB=0, cB=0.
    add_strassen_block(0, 0, 0, 0)
    
    # A01: rA=0, cA=2. B10: rB=2, cB=0.
    add_strassen_block(0, 2, 2, 0)
    
    # A10: rA=2, cA=0. B00: rB=0, cB=0.
    add_strassen_block(2, 0, 0, 0)
    
    # A11: rA=2, cA=2. B10: rB=2, cB=0.
    add_strassen_block(2, 2, 2, 0)
    
    # 4 Schoolbook blocks for C12 and C22 (2x3 outputs)
    # C12 = A00 B01 + A01 B11
    # C22 = A10 B01 + A11 B11
    
    # B01 is 2x3 at rB=0, cB=2.
    # A00: rA=0, cA=0.
    add_schoolbook_223_block(0, 0, 0, 2)
    
    # A01: rA=0, cA=2. B11: rB=2, cB=2.
    add_schoolbook_223_block(0, 2, 2, 2)
    
    # A10: rA=2, cA=0. B01: rB=0, cB=2.
    add_schoolbook_223_block(2, 0, 0, 2)
    
    # A11: rA=2, cA=2. B11: rB=2, cB=2.
    add_schoolbook_223_block(2, 2, 2, 2)
    
    return {"U": U, "V": V, "W": W}
