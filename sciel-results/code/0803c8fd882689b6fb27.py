import numpy as np

def construct():
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: Row 0 of A (1x4) * B (4x4) -> Row 0 of C (1x4)
    # Rank 16
    col = 0
    for k in range(4):
        for j in range(4):
            # U index: row 0, col k -> index k
            U[k, col] = 1
            # V index: row k, col j -> index 4*k + j
            V[4 * k + j, col] = 1
            # W index: row 0, col j -> index j
            W[j, col] = 1
            col += 1
            
    # Part 2: Rows 1-2 of A (2x4) * B (4x4) -> Rows 1-2 of C (2x4)
    # We use 4 Strassen decompositions
    # A11: A[4,5,8,9]
    # A12: A[6,7,10,11]
    # B11: B[0,1,4,5]
    # B12: B[2,3,6,7]
    # B21: B[8,9,12,13]
    # B22: B[10,11,14,15]
    # C21: C[4,5,8,9]
    # C22: C[6,7,10,11]
    
    def add_strassen_block(A_indices, B_indices, C_indices):
        nonlocal col
        a, b, c, d = A_indices
        e, f, g, h = B_indices
        C00, C01, C10, C11 = C_indices
        
        # M1: (a+d)(e+h)
        U[a, col] = 1
        U[d, col] = 1
        V[e, col] = 1
        V[h, col] = 1
        W[C00, col] = 1
        W[C11, col] = 1
        col += 1
        
        # M2: (c+d)e
        U[c, col] = 1
        U[d, col] = 1
        V[e, col] = 1
        W[C10, col] = 1
        W[C11, col] = -1
        col += 1
        
        # M3: a(f-h)
        U[a, col] = 1
        V[f, col] = 1
        V[h, col] = -1
        W[C01, col] = 1
        W[C11, col] = 1
        col += 1
        
        # M4: d(g-e)
        U[d, col] = 1
        V[g, col] = 1
        V[e, col] = -1
        W[C00, col] = 1
        W[C10, col] = 1
        col += 1
        
        # M5: (a+b)h
        U[a, col] = 1
        U[b, col] = 1
        V[h, col] = 1
        W[C00, col] = -1
        W[C01, col] = 1
        col += 1
        
        # M6: (c-d)(f+g)
        U[c, col] = 1
        U[d, col] = -1
        V[f, col] = 1
        V[g, col] = 1
        W[C11, col] = 1
        col += 1
        
        # M7: (b-d)(e+g)
        U[b, col] = 1
        U[d, col] = -1
        V[e, col] = 1
        V[g, col] = 1
        W[C00, col] = 1
        col += 1
        
    # Block 1: A11 * B11 -> C21
    add_strassen_block([4, 5, 8, 9], [0, 1, 4, 5], [4, 5, 8, 9])
    
    # Block 2: A12 * B21 -> C21
    add_strassen_block([6, 7, 10, 11], [8, 9, 12, 13], [4, 5, 8, 9])
    
    # Block 3: A11 * B12 -> C22
    add_strassen_block([4, 5, 8, 9], [2, 3, 6, 7], [6, 7, 10, 11])
    
    # Block 4: A12 * B22 -> C22
    add_strassen_block([6, 7, 10, 11], [10, 11, 14, 15], [6, 7, 10, 11])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
