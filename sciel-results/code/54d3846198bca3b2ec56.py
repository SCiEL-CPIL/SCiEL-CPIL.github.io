import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 68 solution
    # Block 1: i in {0,1}, j in {0,1}, k in {0..7} -> Rank 28
    # Block 2: i in {0,1,2}, j=2 -> Rank 24
    # Block 3: i=2, j in {0,1} -> Rank 16
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: 2x2x8 using 4 Strassen 2x2x2 blocks
    # Strassen coefficients for 2x2x2
    # M1..M7
    # U (i,j) indices: 0,1
    # V (j,k) indices: 0,1 (local)
    # W (k,i) indices: 0,1 (local)
    
    strassen_U = [
        [0, 0, 0, 0, 0, 0, 0], # i=0, j=0
        [0, 0, 1, 0, 1, 0, 0], # i=0, j=1
        [0, 1, 0, 0, 0, -1, 0], # i=1, j=0
        [1, 0, 0, 1, 0, 0, -1]  # i=1, j=1
    ]
    # Note: Strassen formulas:
    # M1 = (A00+A11)(B00+B11) -> U00=1, U11=1; V00=1, V11=1
    # M2 = (A10+A11)B00 -> U10=1, U11=1; V00=1
    # M3 = A00(B10-B11) -> U00=1; V10=1, V11=-1
    # M4 = A11(B01-B00) -> U11=1; V01=1, V00=-1
    # M5 = (A00+A01)B11 -> U00=1, U01=1; V11=1
    # M6 = (A10-A00)(B00+B01) -> U10=1, U00=-1; V00=1, V01=1
    # M7 = (A01-A11)(B10+B11) -> U01=1, U11=-1; V10=1, V11=1
    
    # Correcting U matrix based on standard Strassen:
    # M1: A00+A11, B00+B11
    # M2: A10+A11, B00
    # M3: A00, B10-B11
    # M4: A11, B01-B00
    # M5: A00+A01, B11
    # M6: A10-A00, B00+B01
    # M7: A01-A11, B10+B11
    
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1
        [0, 1, 0, 0, 0, 1, 0],  # i=1, j=0
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1
    ]
    
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0
        [0, 0, 1, 1, 1, 1, 0],  # j=0, k=1
        [1, 0, 1, 0, 0, 0, 1],  # j=1, k=0
        [1, 0, -1, 0, 1, 0, 1]  # j=1, k=1
    ]
    
    strassen_W = [
        [1, 0, 0, -1, 1, 0, 0], # k=0, i=0
        [0, 0, 1, 1, 0, 0, 0],  # k=0, i=1
        [0, 1, 0, 0, 0, 1, 1],  # k=1, i=0
        [1, -1, 0, 0, 0, 1, 0]  # k=1, i=1
    ]
    
    # Fill Block 1
    for q in range(4):
        k_start = 2 * q
        for r_local in range(7):
            r = col + r_local
            # U
            for i in range(2):
                for j in range(2):
                    U[i * m + j][r] = strassen_U[i * 2 + j][r_local]
            # V
            for j in range(2):
                for k_local in range(2):
                    k = k_start + k_local
                    V[j * p + k][r] = strassen_V[j * 2 + k_local][r_local]
            # W
            for k_local in range(2):
                k = k_start + k_local
                for i in range(2):
                    W[k * n + i][r] = strassen_W[k_local * 2 + i][r_local]
            col += 1
            
    # Block 2: i in {0,1,2}, j=2. Rank 24.
    # C_i,k += A_i,2 B_2,k
    # For each i, k, we need a term.
    # U[i, 2, r] = 1
    # V[2, k, r] = 1
    # W[k, i, r] = 1
    for i in range(3):
        for k in range(8):
            r = col
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            col += 1
            
    # Block 3: i=2, j in {0,1}. Rank 16.
    # C_2,k += A_2,j B_j,k
    # For each j, k, we need a term.
    # U[2, j, r] = 1
    # V[j, k, r] = 1
    # W[k, 2, r] = 1
    for j in range(2):
        for k in range(8):
            r = col
            U[2 * m + j][r] = 1
            V[j * p + k][r] = 1
            W[k * n + 2][r] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
