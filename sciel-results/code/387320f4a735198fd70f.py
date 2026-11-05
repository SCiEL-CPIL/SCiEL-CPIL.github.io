"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based rank-49 decomposition for 4x4x4 subproblems
    # plus rank-16 schoolbook for 4x4x1, total rank 114.
    n, m, p = 4, 4, 9
    
    # Rank-7 Strassen decomposition for 2x2x2
    # M1 = (A11+A22)*(B12-B22)
    # M2 = (A11+A22)*(B21+B22)
    # M3 = A11*(B11-B21)
    # M4 = A22*(B22-B12)
    # M5 = (A12+A22)*B22
    # M6 = (A11-A12)*(B11+B12)
    # M7 = (A21-A11)*(B21+B22)
    
    # For 2x2x2:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M1 + M2
    # C22 = M2 - M3 + M4 + M6
    
    # U for 2x2 (A indices): rows are (i,j) pairs for A
    # A has indices (0,0), (0,1), (1,0), (1,1)
    # V for 2x2 (B indices): rows are (j,k) pairs for B
    # B has indices (0,0), (0,1), (1,0), (1,1)
    # W for 2x2 (C indices): rows are (k,i) pairs for C
    # C has indices (0,0), (0,1), (1,0), (1,1)
    
    # Strassen U factors (each of size 4x7, one column per M factor)
    # U[i*2+j, r]
    U2 = [
        # col0: M1 = (A11+A22)*(B12-B22)
        [1, 0, 0, 0, 0, 0, 0],   # A11
        [0, 0, 0, 0, 1, 0, 0],   # A12
        [0, 0, 0, 0, 0, 0, 0],   # A21
        [1, 0, 0, 0, 0, 0, 0],   # A22
        # col1: M2 = (A11+A22)*(B21+B22)
        [1, 0, 0, 0, 0, 0, 0],   # A11
        [0, 0, 0, 0, 0, 0, 0],   # A12
        [0, 0, 0, 0, 0, 0, 0],   # A21
        [1, 0, 0, 0, 0, 0, 0],   # A22
        # col2: M3 = A11*(B11-B21)
        [1, 0, 0, 0, 0, 0, 0],   # A11
        [0, 0, 0, 0, 0, 0, 0],   # A12
        [0, 0, 0, 0, 0, 0, 0],   # A21
        [0, 0, 0, 0, 0, 0, 0],   # A22
        # col3: M4 = A22*(B22-B12)
        [0, 0, 0, 0, 0, 0, 0],   # A11
        [0, 0, 0, 0, 0, 0, 0],   # A12
        [0, 0, 0, 0, 0, 0, 0],   # A21
        [1, 0, 0, 0, 0, 0, 0],   # A22
        # col4: M5 = (A12+A22)*B22
        [0, 0, 0, 0, 0, 0, 0],   # A11
        [1, 0, 0, 0, 1, 0, 0],   # A12
        [0, 0, 0, 0, 0, 0, 0],   # A21
        [1, 0, 0, 0, 1, 0, 0],   # A22
        # col5: M6 = (A11-A12)*(B11+B12)
        [1, 0, 0, 0, 0, 1, 0],   # A11
        [-1, 0, 0, 0, 0, 1, 0],  # A12
        [0, 0, 0, 0, 0, 0, 0],   # A21
        [0, 0, 0, 0, 0, 0, 0],   # A22
        # col6: M7 = (A21-A11)*(B21+B22)
        [-1, 0, 0, 0, 0, 0, 1],  # A11
        [0, 0, 0, 0, 0, 0, 0],   # A12
        [1, 0, 0, 0, 0, 0, 1],   # A21
        [0, 0, 0, 0, 0, 0, 0],   # A22
    ]
    
    # Strassen V factors (each of size 4x7, one column per M factor)
    # V[j*2+k, r]
    V2 = [
        # col0: M1 = (A11+A22)*(B12-B22)
        [0, 0, 0, 0, 0, 0, 0],   # B11
        [1, 0, 0, 0, 0, 0, 0],   # B12
        [0, 0, 0, 0, 0, 0, 0],   # B21
        [-1, 0, 0, 0, 0, 0, 0],  # B22
        # col1: M2 = (A11+A22)*(B21+B22)
        [0, 0, 0, 0, 0, 0, 0],   # B11
        [0, 0, 0, 0, 0, 0, 0],   # B12
        [1, 0, 0, 0, 0, 0, 0],   # B21
        [1, 0, 0, 0, 0, 0, 0],   # B22
        # col2: M3 = A11*(B11-B21)
        [1, 0, 0, 0, 0, 0, 0],   # B11
        [0, 0, 0, 0, 0, 0, 0],   # B12
        [-1, 0, 0, 0, 0, 0, 0],  # B21
        [0, 0, 0, 0, 0, 0, 0],   # B22
        # col3: M4 = A22*(B22-B12)
        [0, 0, 0, 0, 0, 0, 0],   # B11
        [-1, 0, 0, 0, 0, 0, 0],  # B12
        [0, 0, 0, 0, 0, 0, 0],   # B21
        [1, 0, 0, 0, 0, 0, 0],   # B22
        # col4: M5 = (A12+A22)*B22
        [0, 0, 0, 0, 0, 0, 0],   # B11
        [0, 0, 0, 0, 0, 0, 0],   # B12
        [0, 0, 0, 0, 0, 0, 0],   # B21
        [1, 0, 0, 0, 1, 0, 0],   # B22
        # col5: M6 = (A11-A12)*(B11+B12)
        [1, 0, 0, 0, 0, 1, 0],   # B11
        [1, 0, 0, 0, 0, 1, 0],   # B12
        [0, 0, 0, 0, 0, 0, 0],   # B21
        [0, 0, 0, 0, 0, 0, 0],   # B22
        # col6: M7 = (A21-A11)*(B21+B22)
        [0, 0, 0, 0, 0, 0, 0],   # B11
        [0, 0, 0, 0, 0, 0, 0],   # B12
        [1, 0, 0, 0, 0, 0, 1],   # B21
        [1, 0, 0, 0, 0, 0, 1],   # B22
    ]
    
    # Strassen W factors (each of size 4x7, one column per M factor)
    # W[k*2+i, r]
    W2 = [
        # col0: C11 = M1 + M4 - M5 + M7
        [1, 0, 0, 0, -1, 0, 1],  # C11 (k=0,i=0)
        [0, 0, 0, 0, 0, 0, 0],   # C12 (k=0,i=1)
        [0, 0, 0, 0, 0, 0, 0],   # C21 (k=1,i=0)
        [0, 0, 0, 0, 0, 0, 0],   # C22 (k=1,i=1)
        # col1: C21 = M1 + M2
        [0, 0, 0, 0, 0, 0, 0],   # C11
        [0, 0, 0, 0, 0, 0, 0],   # C12
        [1, 1, 0, 0, 0, 0, 0],   # C21
        [0, 0, 0, 0, 0, 0, 0],   # C22
        # col2: C12 = M3 + M5
        [0, 0, 0, 0, 0, 0, 0],   # C11
        [1, 0, 1, 0, 0, 0, 0],   # C12
        [0, 0, 0, 0, 0, 0, 0],   # C21
        [0, 0, 0, 0, 0, 0, 0],   # C22
        # col3: C11 = M4, C22 = M4
        [1, 0, 0, 1, 0, 0, 0],   # C11
        [0, 0, 0, 0, 0, 0, 0],   # C12
        [0, 0, 0, 0, 0, 0, 0],   # C21
        [1, 0, 0, 1, 0, 0, 0],   # C22
        # col4: C11 = -M5, C12 = M5
        [-1, 0, 0, 0, -1, 0, 0], # C11
        [1, 0, 0, 0, 1, 0, 0],   # C12
        [0, 0, 0, 0, 0, 0, 0],   # C21
        [0, 0, 0, 0, 0, 0, 0],   # C22
        # col5: C22 = M6
        [0, 0, 0, 0, 0, 0, 0],   # C11
        [0, 0, 0, 0, 0, 0, 0],   # C12
        [0, 0, 0, 0, 0, 0, 0],   # C21
        [1, 0, 0, 0, 0, 1, 0],   # C22
        # col6: C11 = M7
        [1, 0, 0, 0, 0, 0, 1],   # C11
        [0, 0, 0, 0, 0, 0, 0],   # C12
        [0, 0, 0, 0, 0, 0, 0],   # C21
        [0, 0, 0, 0, 0, 0, 0],   # C22
    ]
    
    # Now build rank-49 decomposition for 4x4x4 using Kronecker products
    # U4[i*4+j, r] where r = r1*7 + r2 (r1, r2 in 0..6)
    # V4[j*4+k, r]
    # W4[k*4+i, r]
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i in range(2):
                for j in range(2):
                    # U4 indices
                    idx1 = i * 2 + j  # position in U2
                    val1 = U2[idx1][r1]
                    idx2 = (i+2) * 2 + j  # wait, this is wrong
                    
                    # Let me reconsider. For Kronecker product:
                    # U4[(i1*2+i2)*(j1*2+j2), r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
                    # Actually, the index mapping is:
                    # A is 4x4, index (I,J) where I = i1*2+i2, J = j1*2+j2
                    # So U4[I*4+J, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
                    
                    pass
    
    # Let me redo this more carefully
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            I = i1 * 2 + i2
                            J = j1 * 2 + j2
                            # U4[I*4+J, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
                            u1 = U2[i1 * 2 + j1][r1]
                            u2 = U2[i2 * 2 + j2][r2]
                            U4[I * 4 + J][r] = u1 * u2
                            
                            # V4[J*4+K, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
                            for k1 in range(2):
                                for k2 in range(2):
                                    K = k1 * 2 + k2
                                    v1 = V2[j1 * 2 + k1][r1]
                                    v2 = V2[j2 * 2 + k2][r2]
                                    V4[J * 4 + K][r] = v1 * v2
                            
                            # W4[K*4+I, r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
                            for k1 in range(2):
                                for k2 in range(2):
                                    K = k1 * 2 + k2
                                    w1 = W2[k1 * 2 + i1][r1]
                                    w2 = W2[k2 * 2 + i2][r2]
                                    W4[K * 4 + I][r] = w1 * w2
    
    # Schoolbook decomposition for 4x4x1, rank 16
    # U[i*4+j, c] = 1 if c = i*4+j
    # V[j*1+k, c] = 1 if c = j*1+0 (only k=0)
    # W[k*4+i, c] = 1 if c = 0*4+i
    
    U1 = [[0]*16 for _ in range(16)]
    V1 = [[0]*16 for _ in range(4)]
    W1 = [[0]*16 for _ in range(4)]
    
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            U1[i * 4 + j][c] = 1
            V1[j][c] = 1
            W1[i][c] = 1
    
    # Now combine: total rank = 49 + 16 + 49 = 114
    # Part 1: cols 0-3 of B (4x4x4), rank 49
    # Part 2: col 4 of B (4x4x1), rank 16
    # Part 3: cols 5-8 of B (4x4x4), rank 49
    
    rank = 114
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(36)]  # m*p = 4*9
    W = [[0]*rank for _ in range(36)]  # p*n = 9*4
    
    # Part 1: cols 0-3 of B
    for i in range(4):
        for j in range(4):
            for r in range(49):
                col = r
                U[i * 4 + j][col] = U4[i * 4 + j][r]
        
        for j in range(4):
            for k in range(4):
                for r in range(49):
                    col = r
                    V[j * 9 + k][col] = V4[j * 4 + k][r]
    
    for k in range(4):
        for i in range(4):
            for r in range(49):
                col = r
                W[k * 4 + i][col] = W4[k * 4 + i][r]
    
    # Part 2: col 4 of B
    for i in range(4):
        for j in range(4):
            for c in range(16):
                col = 49 + c
                U[i * 4 + j][col] = U1[i * 4 + j][c]
    
    for j in range(4):
        for c in range(16):
            col = 49 + c
            V[j * 9 + 4][col] = V1[j][c]
    
    for i in range(4):
        for c in range(16):
            col = 49 + c
            W[4 * 4 + i][col] = W1[i][c]
    
    # Part 3: cols 5-8 of B
    for i in range(4):
        for j in range(4):
            for r in range(49):
                col = 49 + 16 + r
                U[i * 4 + j][col] = U4[i * 4 + j][r]
    
    for j in range(4):
        for k in range(4):
            for r in range(49):
                col = 49 + 16 + r
                V[j * 9 + (k + 5)][col] = V4[j * 4 + k][r]
    
    for k in range(4):
        for i in range(4):
            for r in range(49):
                col = 49 + 16 + r
                W[(k + 5) * 4 + i][col] = W4[k * 4 + i][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
