import numpy as np

def construct():
    n, m, p = 3, 3, 8
    rank = 68
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a term
    def add_term(U_idx, V_idx, W_idx, coeff=1):
        nonlocal col
        U[U_idx, col] += coeff
        V[V_idx, col] += coeff
        W[W_idx, col] += coeff
        col += 1

    # Loop over 4 blocks of B (each 2 columns)
    for j in range(4):
        b_start = 2 * j
        b_end = 2 * j + 2
        cols = [b_start, b_start + 1]
        
        # Block 1: A[0:2, 0:2] * B[0:2, cols]
        # A indices: (0,0), (0,1), (1,0), (1,1)
        # B indices: (0, c), (1, c) for c in cols
        
        # M1 = (A00 + A11)(B00 + B11)
        add_term([0, 3], [cols[0], 3 + cols[0]], [cols[0] * 3 + 0, cols[1] * 3 + 1]) # Wait, W indices need to be specific
        
        # W index is k*n + i.
        # For C[0,0] (i=0, k=cols[0]), W index is cols[0]*3 + 0.
        # For C[1,1] (i=1, k=cols[1]), W index is cols[1]*3 + 1.
        # This term contributes to multiple C entries.
        # We must add the coefficient to each W entry.
        
        # Reset and do it properly
        
        # Block 1: Strassen
        # A00=U[0], A01=U[1], A10=U[3], A11=U[4]
        # B00=V[0*8+c0], B01=V[0*8+c1], B10=V[1*8+c0], B11=V[1*8+c1]
        c0, c1 = cols
        
        # M1
        U[0, col] += 1; U[4, col] += 1
        V[c0, col] += 1; V[8+c0, col] += 1
        W[c0*3, col] += 1     # C[0,0]
        W[c1*3+1, col] += 1   # C[1,1]
        col += 1
        
        # M2
        U[3, col] += 1; U[4, col] += 1
        V[c0, col] += 1
        W[c0*3+1, col] += 1   # C[1,0]
        W[c1*3+1, col] += -1  # C[1,1]
        col += 1
        
        # M3
        U[0, col] += 1
        V[8+c1, col] += 1; V[c0, col] += -1
        W[c1*3, col] += 1     # C[0,1]
        W[c1*3+1, col] += 1   # C[1,1]
        col += 1
        
        # M4
        U[4, col] += 1
        V[c0, col] += 1; V[8+c0, col] += 1
        W[c0*3, col] += 1     # C[0,0]
        W[c1*3+1, col] += 1   # C[1,1]
        col += 1
        
        # M5
        U[0, col] += 1; U[3, col] += -1
        V[8+c0, col] += 1; V[8+c1, col] += 1
        W[c0*3, col] += -1    # C[0,0]
        W[c0*3+1, col] += 1   # C[1,0]
        col += 1
        
        # M6
        U[1, col] += 1; U[4, col] += -1
        V[c0, col] += 1; V[8+c0, col] += 1
        W[c1*3, col] += 1     # C[0,1]
        col += 1
        
        # M7
        U[0, col] += 1; U[1, col] += 1
        V[8+c1, col] += 1
        W[c0*3, col] += -1    # C[0,0]
        W[c1*3, col] += 1     # C[0,1]
        col += 1
        
        # Block 2: A[0:2, 2] * B[2, cols]
        # A02=U[2], A12=U[5]
        # B20=V[2*8+c0], B21=V[2*8+c1]
        
        # M1
        U[2, col] += 1
        V[16+c0, col] += 1
        W[c0*3, col] += 1
        col += 1
        
        # M2
        U[2, col] += 1
        V[16+c1, col] += 1
        W[c1*3, col] += 1
        col += 1
        
        # M3
        U[5, col] += 1
        V[16+c0, col] += 1
        W[c0*3+1, col] += 1
        col += 1
        
        # M4
        U[5, col] += 1
        V[16+c1, col] += 1
        W[c1*3+1, col] += 1
        col += 1
        
        # Block 3: A[2, 0:2] * B[0:2, cols]
        # A20=U[6], A21=U[7]
        
        # M1
        U[6, col] += 1
        V[c0, col] += 1
        W[c0*3+2, col] += 1
        col += 1
        
        # M2
        U[6, col] += 1
        V[8+c0, col] += 1
        W[c1*3+2, col] += 1
        col += 1
        
        # M3
        U[7, col] += 1
        V[c0, col] += 1
        W[c0*3+2, col] += 1
        col += 1
        
        # M4
        U[7, col] += 1
        V[8+c0, col] += 1
        W[c1*3+2, col] += 1
        col += 1
        
        # Block 4: A[2, 2] * B[2, cols]
        # A22=U[8]
        
        # M1
        U[8, col] += 1
        V[16+c0, col] += 1
        W[c0*3+2, col] += 1
        col += 1
        
        # M2
        U[8, col] += 1
        V[16+c1, col] += 1
        W[c1*3+2, col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
