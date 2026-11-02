"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U2[i, c], V2[j, c], W2[k, c] for i,j,k in 0..1, c in 0..6
    # Indices:
    # c=0: M11*M22 -> (A+B)(C+D)
    # c=1: M11*M12 -> (A+B)(B-D)
    # c=2: M21*M11 -> (A+C)(A+B)
    # c=3: M22*M21 -> (C+D)(A+C)
    # c=4: M12*M21 -> (B-D)(A+C)
    # c=5: (M11+M22)*M12 -> (A+C)(B-D) ? No, (A+D)(B-C)? 
    # Let's use the standard Strassen formulas:
    # P1 = (A+B)(C+D)
    # P2 = (A+B)(C-D)
    # P3 = (A+C)(B)
    # P4 = (D-B)(A+C)
    # P5 = (A)(D)
    # P6 = (B)(C-D)
    # P7 = (A-C)(D+B)
    # X = P1 + P4 - P6 + P7
    # Y = P3 + P5
    # Z = P1 - P2 + P5 + P6
    # W = P1 + P4 - P3 + P5
    
    # This corresponds to:
    # U2 rows: A, B, C, D (indices 0,1,2,3)
    # V2 rows: A, B, C, D (indices 0,1,2,3)
    # W2 rows: X, Y, Z, W (indices 0,1,2,3)
    
    U2 = [
        [1, 1, 1, 0, 1, 0, 1], # A
        [1, 1, 0, 1, 0, 1, 0], # B
        [0, 0, 1, 1, 1, 0, -1], # C
        [1, 0, 0, 1, 0, 0, 1]  # D
    ]
    V2 = [
        [1, 0, 0, 1, 0, 0, 1], # A
        [0, 1, 1, 0, 1, 1, 0], # B
        [1, 1, 0, 0, 0, 0, 0], # C
        [0, 0, 1, 1, 0, 1, 1]  # D
    ]
    W2 = [
        [1, 0, 0, 1, 0, 0, 1], # X
        [0, 0, 1, 0, 1, 0, 0], # Y
        [1, -1, 0, 0, 1, 1, 0], # Z
        [1, 0, -1, 1, 1, 0, 0]  # W
    ]
    
    # Verify Strassen:
    # P1: (A+B)(C+D) -> U: A:1, B:1; V: C:1, D:1; W: X:1
    # P2: (A+B)(C-D) -> U: A:1, B:1; V: C:1, D:-1; W: X:0, Z:-1
    # P3: (A+C)(B) -> U: A:1, C:1; V: B:1; W: Y:1
    # P4: (D-B)(A+C) -> U: D:1, B:-1; V: A:1, C:1; W: X:1, W:1
    # P5: (A)(D) -> U: A:1; V: D:1; W: Y:1, Z:1, W:1
    # P6: (B)(C-D) -> U: B:1; V: C:1, D:-1; W: X:-1, Z:1
    # P7: (A-C)(D+B) -> U: A:1, C:-1; V: D:1, B:1; W: X:1, W:1
    
    # My U2, V2, W2 above:
    # c=0: U: A:1, B:1, C:0, D:1. V: A:1, B:0, C:1, D:0. W: X:1, Y:0, Z:1, W:1.
    # This is (A+B+D)(A+C)? No.
    
    # Let's use the explicit 7 terms from a known source to be safe.
    # Strassen 1969:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C = M1 + M4 - M5 + M7
    # D = M3 + M5
    # E = M2 + M4
    # F = M1 - M2 + M3 + M6
    
    # Let's map this to U, V, W.
    # U rows: A11, A12, A21, A22 (indices 0,1,2,3)
    # V rows: B11, B12, B21, B22 (indices 0,1,2,3)
    # W rows: C, D, E, F (indices 0,1,2,3)
    
    U2 = [
        [1, 0, 1, 0, 1, -1, -1], # A11
        [0, 0, 0, 0, 1, 0, 1],   # A12
        [0, 1, 0, 0, 0, 1, 0],   # A21
        [1, 1, 0, 1, 0, 0, -1]   # A22
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0],   # B11
        [0, 0, 0, 1, 0, 1, 0],   # B12
        [1, 0, 1, 0, 0, 0, 1],   # B21
        [1, 0, -1, -1, 1, 0, 1]  # B22
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # C
        [0, 0, 1, 0, 1, 0, 0],   # D
        [0, 1, 0, 1, 0, 0, 0],   # E
        [1, -1, 1, 0, 0, 1, 0]   # F
    ]
    
    # Now build 4x4x4 using tensor power.
    # Rank 49.
    U444 = [[0]*49 for _ in range(16)]
    V444 = [[0]*49 for _ in range(16)]
    W444 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for i2 in range(2):
            i = i1*2 + i2
            for j1 in range(2):
                for j2 in range(2):
                    j = j1*2 + j2
                    for k1 in range(2):
                        for k2 in range(2):
                            k = k1*2 + k2
                            # This is not how tensor power works directly on indices.
                            # Tensor power:
                            # U444[i, c] where c = c1*7 + c2
                            # U444[i1*2+i2, c1*7+c2] = U2[i1, c1] * U2[i2, c2]
                            pass
    
    # Correct construction:
    for i in range(4):
        i1, i2 = i // 2, i % 2
        for c in range(49):
            c1, c2 = c // 7, c % 7
            U444[i][c] = U2[i1][c1] * U2[i2][c2]
            
    for j in range(4):
        j1, j2 = j // 2, j % 2
        for c in range(49):
            c1, c2 = c // 7, c % 7
            V444[j][c] = V2[j1][c1] * V2[j2][c2]
            
    for k in range(4):
        k1, k2 = k // 2, k % 2
        for c in range(49):
            c1, c2 = c // 7, c % 7
            W444[k][c] = W2[k1][c1] * W2[k2][c2]
            
    # Now build 4x4x8.
    # Rank 98.
    # U: 16 x 98
    # V: 32 x 98
    # W: 32 x 98
    
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)]
    W = [[0]*98 for _ in range(32)]
    
    # First block (k=0..3)
    for i in range(16):
        for c in range(49):
            U[i][c] = U444[i][c]
            
    for j in range(16):
        for c in range(49):
            V[j][c] = V444[j][c]
            
    for k in range(16):
        for c in range(49):
            W[k][c] = W444[k][c]
            
    # Second block (k=4..7)
    for i in range(16):
        for c in range(49, 98):
            U[i][c] = U444[i][c-49]
            
    for j in range(16, 32):
        for c in range(49, 98):
            V[j][c] = V444[j-16][c-49]
            
    for k in range(16, 32):
        for c in range(49, 98):
            W[k][c] = W444[k-16][c-49]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
