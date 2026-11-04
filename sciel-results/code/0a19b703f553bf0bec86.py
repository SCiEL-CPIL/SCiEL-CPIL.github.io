import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Rank 39 decomposition
    # Structure:
    # 1. Three 2x2x2 Strassen blocks for A[0:2, 0:2] * B[0:2, 0:6]
    # 2. One 2x2x1 Schoolbook block for A[0:2, 0:2] * B[0:2, 6]
    # 3. One 2x1x7 Schoolbook block for A[0:2, 2] * B[2, 0:7]
    
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    def add_strassen_block(k_offset):
        nonlocal col
        # B indices for rows 0, 1 and cols k_offset, k_offset+1
        b00 = 0 * p + k_offset
        b01 = 0 * p + k_offset + 1
        b10 = 1 * p + k_offset
        b11 = 1 * p + k_offset + 1
        
        # C indices for cols k_offset, k_offset+1 and rows 0, 1
        c00 = k_offset * n + 0
        c01 = (k_offset + 1) * n + 0
        c10 = k_offset * n + 1
        c11 = (k_offset + 1) * n + 1
        
        # A indices for rows 0, 1 and cols 0, 1
        a00 = 0 * m + 0
        a01 = 0 * m + 1
        a10 = 1 * m + 0
        a11 = 1 * m + 1
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10, -C11
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        W[c11, col] += -1
        col += 1
        
        # M3: A00(B01-B11) -> C01, C11
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] += -1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M4: A11(B10-B00) -> C00, C10
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # M5: (A00+A01)B11 -> -C00, C01
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1
        W[c01, col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U[a10, col] += 1
        U[a00, col] += -1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00
        U[a01, col] += 1
        U[a11, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1

    # Apply Strassen blocks for k=0, 2, 4
    for k in [0, 2, 4]:
        add_strassen_block(k)
        
    # Schoolbook block for k=6, rows 0,1 of A and B
    # C[i, 6] = A[i, 0] B[0, 6] + A[i, 1] B[1, 6]
    # This is 4 terms
    b06 = 0 * p + 6
    b16 = 1 * p + 6
    c06 = 6 * n + 0
    c16 = 6 * n + 1
    
    # Term 1: A[0,0] B[0,6] C[0,6]
    U[0 * m + 0, col] = 1
    V[b06, col] = 1
    W[c06, col] = 1
    col += 1
    
    # Term 2: A[1,0] B[1,6] C[0,6]
    U[1 * m + 0, col] = 1
    V[b16, col] = 1
    W[c06, col] = 1
    col += 1
    
    # Term 3: A[0,1] B[0,6] C[1,6]
    U[0 * m + 1, col] = 1
    V[b06, col] = 1
    W[c16, col] = 1
    col += 1
    
    # Term 4: A[1,1] B[1,6] C[1,6]
    U[1 * m + 1, col] = 1
    V[b16, col] = 1
    W[c16, col] = 1
    col += 1
    
    # Schoolbook block for row 2 of A
    # C[i, k] = A[i, 2] B[2, k]
    # 14 terms
    for k in range(p):
        b2k = 2 * p + k
        c0k = k * n + 0
        c1k = k * n + 1
        
        # Term: A[0,2] B[2,k] C[0,k]
        U[0 * m + 2, col] = 1
        V[b2k, col] = 1
        W[c0k, col] = 1
        col += 1
        
        # Term: A[1,2] B[2,k] C[1,k]
        U[1 * m + 2, col] = 1
        V[b2k, col] = 1
        W[c1k, col] = 1
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
