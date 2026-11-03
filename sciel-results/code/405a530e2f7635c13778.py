import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We aim for rank 52
    rank = 52
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    # A indices: i in [0,1], j in [j0, j0+1]
    # B indices: j in [j0, j0+1], k in [k0, k0+1]
    # C indices: i in [0,1], k in [k0, k0+1]
    def add_strassen_222(j0, k0):
        nonlocal col
        # m1 = (a00+a01)(b00+b11)
        U[0*m + 0 + j0][col] += 1
        U[0*m + 1 + j0][col] += 1
        V[0*p + k0][col] += 1
        V[1*p + k0+1][col] += 1
        W[k0*n + 0][col] += 1
        W[(k0+1)*n + 1][col] += 1
        W[(k0+1)*n + 0][col] += 1
        W[k0*n + 1][col] += 1 # z00 gets m1, z11 gets m1, z01 gets? No.
        # z00 = m1 + m4 - m5 + m7
        # z11 = m1 - m2 + m3 + m6
        # z01 = m3 + m5
        # z10 = m2 + m4
        # So for m1:
        # W[z00] += 1, W[z11] += 1
        # z00 index: k0*n + 0
        # z11 index: (k0+1)*n + 1
        # z01 index: (k0+1)*n + 0 ? No, k is col index.
        # C is 2x2. C[i, k].
        # z00 is i=0, k=k0. Index k0*n + 0.
        # z11 is i=1, k=k0+1. Index (k0+1)*n + 1.
        # z01 is i=0, k=k0+1. Index (k0+1)*n + 0.
        # z10 is i=1, k=k0. Index k0*n + 1.
        
        # Correct W updates for m1:
        # W[k0*n + 0] += 1
        # W[(k0+1)*n + 1] += 1
        # (I already added these above, but let's clear and redo carefully)
        
        # Reset W for this col
        W[:, col] = 0
        
        # m1
        U[0*m + 0 + j0][col] = 1
        U[0*m + 1 + j0][col] = 1
        V[0*p + k0][col] = 1
        V[1*p + k0+1][col] = 1
        W[k0*n + 0][col] = 1
        W[(k0+1)*n + 1][col] = 1
        col += 1
        
        # m2 = (a10+a11)b11
        U[1*m + 0 + j0][col] = 1
        U[1*m + 1 + j0][col] = 1
        V[1*p + k0+1][col] = 1
        W[k0*n + 1][col] = 1
        W[(k0+1)*n + 1][col] = -1
        col += 1
        
        # m3 = a00(b01-b10)
        U[0*m + 0 + j0][col] = 1
        V[0*p + k0+1][col] = 1
        V[1*p + k0][col] = -1
        W[(k0+1)*n + 0][col] = 1
        W[(k0+1)*n + 1][col] = 1
        col += 1
        
        # m4 = a11(b00-b10)
        U[1*m + 1 + j0][col] = 1
        V[0*p + k0][col] = 1
        V[1*p + k0][col] = -1
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        
        # m5 = (a00+a01)b01
        U[0*m + 0 + j0][col] = 1
        U[0*m + 1 + j0][col] = 1
        V[0*p + k0+1][col] = 1
        W[(k0+1)*n + 0][col] = -1
        W[(k0+1)*n + 1][col] = 1
        col += 1
        
        # m6 = (a10-a00)(b00+b10)
        U[1*m + 0 + j0][col] = 1
        U[0*m + 0 + j0][col] = -1
        V[0*p + k0][col] = 1
        V[1*p + k0][col] = 1
        W[(k0+1)*n + 1][col] = 1
        col += 1
        
        # m7 = (a01-a11)(b10+b11)
        U[0*m + 1 + j0][col] = 1
        U[1*m + 1 + j0][col] = -1
        V[1*p + k0][col] = 1
        V[1*p + k0+1][col] = 1
        W[k0*n + 0][col] = 1
        W[(k0+1)*n + 0][col] = 1
        col += 1
        
    # Helper to add Schoolbook 2x2x3
    # A indices: i in [0,1], j in [j0, j0+1]
    # B indices: j in [j0, j0+1], k in [k0, k0+2]
    # C indices: i in [0,1], k in [k0, k0+2]
    def add_schoolbook_223(j0, k0):
        nonlocal col
        # 12 terms: a_ij * b_jk
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[i*m + j0 + j][col] = 1
                    V[(j0 + j)*p + k0 + k][col] = 1
                    W[(k0 + k)*n + i][col] = 1
                    col += 1

    # Block 1: A_left (j=0,1) * B_top (j=0,1)
    # B_top is k=0..6. Split into k=0..3 (2x4) and k=4..6 (2x3).
    # 2x4 split into two 2x2: k=0,1 and k=2,3.
    add_strassen_222(0, 0) # j=0,1, k=0,1
    add_strassen_222(0, 2) # j=0,1, k=2,3
    add_schoolbook_223(0, 4) # j=0,1, k=4,5,6
    
    # Block 2: A_right (j=2,3) * B_bottom (j=2,3)
    # B_bottom is k=0..6. Split into k=0..3 and k=4..6.
    add_strassen_222(2, 0) # j=2,3, k=0,1
    add_strassen_222(2, 2) # j=2,3, k=2,3
    add_schoolbook_223(2, 4) # j=2,3, k=4,5,6
    
    # Convert to lists for output
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
