import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Define 2x2 Strassen factors
    # U2, V2, W2 are 4x7 matrices
    # Rows are flattened indices: i*2 + j for U, j*2 + k for V, k*2 + i for W
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1 # A00, A11
    V2[0, 0] = 1; V2[3, 0] = 1 # B00, B11
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1 # A10, A11
    V2[0, 1] = 1 # B00
    W2[2, 1] = 1; W2[3, 1] = -1 # C10, C11
    
    # M3 = A00(B10-B11)
    U2[0, 2] = 1 # A00
    V2[2, 2] = 1; V2[3, 2] = -1 # B10, B11
    W2[1, 2] = 1; W2[3, 2] = 1 # C01, C11
    
    # M4 = A11(B01-B00)
    U2[3, 3] = 1 # A11
    V2[1, 3] = 1; V2[0, 3] = -1 # B01, B00
    W2[0, 3] = 1; W2[2, 3] = 1 # C00, C10
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1 # A00, A01
    V2[3, 4] = 1 # B11
    W2[0, 4] = -1; W2[1, 4] = 1 # C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1 # A10, A00
    V2[0, 5] = 1; V2[1, 5] = 1 # B00, B01
    W2[3, 5] = 1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1 # A01, A11
    V2[2, 6] = 1; V2[3, 6] = 1 # B10, B11
    W2[0, 6] = 1 # C00
    
    # 2. Build 4x4 Strassen factors (Rank 49)
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    # U4[i*4+j, c1*7+c0] = U2[r1, c1] * U2[r0, c0]
    # where i = 2*i1 + i0, j = 2*j1 + j0
    # r1 = 2*i1 + j1, r0 = 2*i0 + j0
    for i in range(4):
        i1 = i // 2; i0 = i % 2
        for j in range(4):
            j1 = j // 2; j0 = j % 2
            r1 = 2 * i1 + j1
            r0 = 2 * i0 + j0
            for c1 in range(7):
                for c0 in range(7):
                    c = c1 * 7 + c0
                    U4[i * 4 + j, c] = U2[r1, c1] * U2[r0, c0]
                    
    # V4[j*4+k, c1*7+c0] = V2[r1, c1] * V2[r0, c0]
    # where j = 2*j1 + j0, k = 2*k1 + k0
    # r1 = 2*j1 + k1, r0 = 2*j0 + k0
    for j in range(4):
        j1 = j // 2; j0 = j % 2
        for k in range(4):
            k1 = k // 2; k0 = k % 2
            r1 = 2 * j1 + k1
            r0 = 2 * j0 + k0
            for c1 in range(7):
                for c0 in range(7):
                    c = c1 * 7 + c0
                    V4[j * 4 + k, c] = V2[r1, c1] * V2[r0, c0]
                    
    # W4[k*4+i, c1*7+c0] = W2[r1, c1] * W2[r0, c0]
    # where k = 2*k1 + k0, i = 2*i1 + i0
    # r1 = 2*k1 + i1, r0 = 2*k0 + i0
    for k in range(4):
        k1 = k // 2; k0 = k % 2
        for i in range(4):
            i1 = i // 2; i0 = i % 2
            r1 = 2 * k1 + i1
            r0 = 2 * k0 + i0
            for c1 in range(7):
                for c0 in range(7):
                    c = c1 * 7 + c0
                    W4[k * 4 + i, c] = W2[r1, c1] * W2[r0, c0]
                    
    # 3. Assemble 5x5 factors
    # Rank = 49 (4x4 part) + 61 (cross terms) = 110
    rank = 110
    U = np.zeros((25, rank), dtype=float)
    V = np.zeros((25, rank), dtype=float)
    W = np.zeros((25, rank), dtype=float)
    
    # Copy 4x4 part into top-left
    # U4 row i*4+j -> U row i*5+j
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, :49] = U4[i * 4 + j, :]
    for j in range(4):
        for k in range(4):
            V[j * 5 + k, :49] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k * 5 + i, :49] = W4[k * 4 + i, :]
            
    c = 49
    # Cross terms
    # 1. i<4, j=4, k<4
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4, c] = 1
            V[4 * 5 + k, c] = 1
            W[k * 5 + i, c] = 1
            c += 1
            
    # 2. i<4, j<4, k=4
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, c] = 1
            V[j * 5 + 4, c] = 1
            W[4 * 5 + i, c] = 1
            c += 1
            
    # 3. i=4, j<4, k<4
    for j in range(4):
        for k in range(4):
            U[4 * 5 + j, c] = 1
            V[j * 5 + k, c] = 1
            W[k * 5 + 4, c] = 1
            c += 1
            
    # 4. i<4, j=4, k=4
    for i in range(4):
        U[i * 5 + 4, c] = 1
        V[4 * 5 + 4, c] = 1
        W[4 * 5 + i, c] = 1
        c += 1
        
    # 5. i=4, j<4, k=4
    for j in range(4):
        U[4 * 5 + j, c] = 1
        V[j * 5 + 4, c] = 1
        W[4 * 5 + 4, c] = 1
        c += 1
        
    # 6. i=4, j=4, k<4
    for k in range(4):
        U[4 * 5 + 4, c] = 1
        V[4 * 5 + 4, c] = 1
        W[k * 5 + 4, c] = 1
        c += 1
        
    # 7. i=4, j=4, k=4
    U[4 * 5 + 4, c] = 1
    V[4 * 5 + 4, c] = 1
    W[4 * 5 + 4, c] = 1
    c += 1
    
    assert c == rank, f"Rank mismatch: {c} vs {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
