import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition of rank 94:
    # 49 from Strassen on 4x4 sub-block
    # 45 from schoolbook on the cross terms involving the 5th row/col
    
    # 1. Strassen 2x2 factors
    # U_S, V_S, W_S are 4x7 matrices
    U_S = np.zeros((4, 7), dtype=float)
    V_S = np.zeros((4, 7), dtype=float)
    W_S = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    U_S[0, 0] = 1; U_S[3, 0] = 1
    V_S[0, 0] = 1; V_S[3, 0] = 1
    W_S[0, 0] = 1; W_S[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U_S[2, 1] = 1; U_S[3, 1] = 1
    V_S[0, 1] = 1
    W_S[2, 1] = 1; W_S[3, 1] = -1
    
    # M3 = A11(B21-B22)
    U_S[0, 2] = 1
    V_S[2, 2] = 1; V_S[3, 2] = -1
    W_S[1, 2] = 1; W_S[3, 2] = 1
    
    # M4 = A22(B12-B11)
    U_S[3, 3] = 1
    V_S[1, 3] = 1; V_S[0, 3] = -1
    W_S[0, 3] = 1; W_S[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U_S[0, 4] = 1; U_S[1, 4] = 1
    V_S[3, 4] = 1
    W_S[0, 4] = -1; W_S[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U_S[2, 5] = 1; U_S[0, 5] = -1
    V_S[0, 5] = 1; V_S[1, 5] = 1
    W_S[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U_S[1, 6] = 1; U_S[3, 6] = -1
    V_S[2, 6] = 1; V_S[3, 6] = 1
    W_S[0, 6] = 1

    # 2. Expand to 4x4 (Rank 49)
    # Indices i, j in 0..3. i = 2*i2 + i1
    U_44 = np.zeros((16, 49), dtype=float)
    V_44 = np.zeros((16, 49), dtype=float)
    W_44 = np.zeros((16, 49), dtype=float)
    
    for i in range(4):
        i1 = i % 2
        i2 = i // 2
        for j in range(4):
            j1 = j % 2
            j2 = j // 2
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    U_44[i * 4 + j, c] = U_S[i1, c1] * U_S[i2, c2]
                    V_44[j * 4 + i, c] = V_S[j1, c1] * V_S[j2, c2] # Note: V index is j*p+k, here p=4 for sub-block
                    # Wait, V index for 4x4 is j*4+k.
                    # In the 2x2 Strassen, V index is j*2+k.
                    # The mapping j -> j1, j2 is consistent.
                    W_44[i * 4 + j, c] = W_S[i1, c1] * W_S[i2, c2] # W index k*n+i. Here k=i, n=4.
                    # Wait, W index in 2x2 is k*2+i.
                    # In 4x4, W index is k*4+i.
                    # The term is W[k, i].
                    # In Strassen, W_S[k, i].
                    # So W_44[k*4+i, c] = W_S[k1, c1] * W_S[k2, c2].
                    # My loop variable i is used for both row of U and col of W?
                    # U[i, j], V[j, k], W[k, i].
                    # In the 4x4 expansion, the indices are i, j, k in 0..3.
                    # U_44[i*4+j, c] = U_S[i1, j1, c1] * U_S[i2, j2, c2]?
                    # No, U_S is for 2x2. U_S[row, col, c].
                    # U_44[i, j, c1, c2] = U_S[i1, j1, c1] * U_S[i2, j2, c2].
                    # This is correct.
                    
                    # Correction on V and W indices in the loop:
                    # I am iterating i (row of U) and j (col of U).
                    # But V depends on j and k. W depends on k and i.
                    # I need to iterate k as well?
                    # No, the tensor product structure means:
                    # U_44[i, j, (c1,c2)] = U_S[i1, j1, c1] * U_S[i2, j2, c2]
                    # V_44[j, k, (c1,c2)] = V_S[j1, k1, c1] * V_S[j2, k2, c2]
                    # W_44[k, i, (c1,c2)] = W_S[k1, i1, c1] * W_S[k2, i2, c2]
                    
                    # I need to fill U, V, W for all i, j, k.
    
    # Re-doing the fill properly
    U_44 = np.zeros((16, 49), dtype=float)
    V_44 = np.zeros((16, 49), dtype=float)
    W_44 = np.zeros((16, 49), dtype=float)
    
    for i in range(4):
        i1 = i % 2; i2 = i // 2
        for j in range(4):
            j1 = j % 2; j2 = j // 2
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    U_44[i * 4 + j, c] = U_S[i1, c1] * U_S[i2, c2] # Wait, U_S is 2x2, indices 0..1.
                    # U_S[row, col, c].
                    # So U_S[i1, j1, c1] * U_S[i2, j2, c2].
                    U_44[i * 4 + j, c] = U_S[i1, j1, c1] * U_S[i2, j2, c2]
    
    for j in range(4):
        j1 = j % 2; j2 = j // 2
        for k in range(4):
            k1 = k % 2; k2 = k // 2
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    V_44[j * 4 + k, c] = V_S[j1, k1, c1] * V_S[j2, k2, c2]
                    
    for k in range(4):
        k1 = k % 2; k2 = k // 2
        for i in range(4):
            i1 = i % 2; i2 = i // 2
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    W_44[k * 4 + i, c] = W_S[k1, i1, c1] * W_S[k2, i2, c2]

    # 3. Add cross terms (Schoolbook)
    # Total rank so far: 49
    # Cross terms:
    # i in 0..3, j=4, k in 0..3 (16 terms)
    # i in 0..3, j in 0..3, k=4 (16 terms)
    # i=4, j in 0..3, k in 0..3 (16 terms)
    # i in 0..3, j=4, k=4 (1 term)
    # i=4, j in 0..3, k=4 (4 terms)
    # i=4, j=4, k in 0..3 (4 terms)
    # i=4, j=4, k=4 (1 term)
    # Total cross: 16+16+16+1+4+4+1 = 58?
    # Wait, my previous count was 45.
    # Let's re-count.
    # A_{4x4} B_{4x4}: 49
    # A_{4x4} b_{5x1}: 16
    # a_{5x1} b_{5,5}: 4
    # a_{1x4} B_{4x4}: 16
    # a_{5,5} b_{1x4}: 4
    # a_{1x4} b_{5x1}: 4
    # a_{5,5} b_{5,5}: 1
    # Sum: 49 + 16 + 4 + 16 + 4 + 4 + 1 = 94.
    
    rank = 94
    U = np.zeros((25, rank), dtype=float)
    V = np.zeros((25, rank), dtype=float)
    W = np.zeros((25, rank), dtype=float)
    
    # Copy 4x4 part
    U[:, :49] = 0
    V[:, :49] = 0
    W[:, :49] = 0
    # U_44 is 16x49. Rows i*4+j for i,j in 0..3.
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, :49] = U_44[i * 4 + j, :]
    for j in range(4):
        for k in range(4):
            V[j * 5 + k, :49] = V_44[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k * 5 + i, :49] = W_44[k * 4 + i, :]

    # Add cross terms
    c = 49
    # i in 0..3, j=4, k in 0..3
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4, c] = 1
            V[4 * 5 + k, c] = 1
            W[k * 5 + i, c] = 1
            c += 1
            
    # i in 0..3, j in 0..3, k=4
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, c] = 1
            V[j * 5 + 4, c] = 1
            W[4 * 5 + i, c] = 1
            c += 1
            
    # i=4, j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            U[4 * 5 + j, c] = 1
            V[j * 5 + k, c] = 1
            W[k * 5 + 4, c] = 1
            c += 1
            
    # i in 0..3, j=4, k=4
    for i in range(4):
        U[i * 5 + 4, c] = 1
        V[4 * 5 + 4, c] = 1
        W[4 * 5 + i, c] = 1
        c += 1
        
    # i=4, j in 0..3, k=4
    for j in range(4):
        U[4 * 5 + j, c] = 1
        V[j * 5 + 4, c] = 1
        W[4 * 5 + 4, c] = 1
        c += 1
        
    # i=4, j=4, k in 0..3
    for k in range(4):
        U[4 * 5 + 4, c] = 1
        V[4 * 5 + 4, c] = 1
        W[k * 5 + 4, c] = 1
        c += 1
        
    # i=4, j=4, k=4
    U[4 * 5 + 4, c] = 1
    V[4 * 5 + 4, c] = 1
    W[4 * 5 + 4, c] = 1
    c += 1
    
    assert c == rank, f"Rank mismatch: {c} vs {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
