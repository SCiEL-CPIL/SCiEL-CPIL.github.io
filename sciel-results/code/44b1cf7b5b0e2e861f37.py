"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition of rank 28 for 2x4 * 4x4 multiplication.
    # We split A (2x4) into two 2x2 blocks A1, A2 and B (4x4) into four 2x2 blocks B11, B12, B21, B22.
    # C = A1*B11 + A2*B21 (top-left 2x2) + A1*B12 + A2*B22 (top-right 2x2)
    # Actually, C[i,k] = sum_j A[i,j] B[j,k]
    # C[0:2, 0:2] = A[0:2, 0:2] @ B[0:2, 0:2] + A[0:2, 2:4] @ B[2:4, 0:2]
    # C[0:2, 2:4] = A[0:2, 0:2] @ B[0:2, 2:4] + A[0:2, 2:4] @ B[2:4, 2:4]
    
    # We use 4 independent Strassen multiplications:
    # S1: A1 @ B11 -> contributes to C[0:2, 0:2]
    # S2: A1 @ B12 -> contributes to C[0:2, 2:4]
    # S3: A2 @ B21 -> contributes to C[0:2, 0:2]
    # S4: A2 @ B22 -> contributes to C[0:2, 2:4]
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    
    # U: shape (n*m, rank) = (8, 28)
    # V: shape (m*p, rank) = (16, 28)
    # W: shape (p*n, rank) = (8, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper: Strassen for 2x2 * 2x2
    # Inputs: A (2x2), B (2x2)
    # Indices in A: a00, a01, a10, a11
    # Indices in B: b00, b01, b10, b11
    # Multiplications:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b10-b11)
    # m4 = a11(b01-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    # Output:
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    def strassen_coeffs():
        # Returns list of 7 tuples (u_vec, v_vec, w_vec) where each is a dict from local index to coefficient
        # Local indices for A (2x2): (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # Local indices for B (2x2): (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # Local indices for C (2x2): (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        
        def vec(d):
            return [d.get(i, 0) for i in range(4)]
        
        # m1 = (a00+a11)(b00+b11)
        u1 = vec({0: 1, 3: 1})
        v1 = vec({0: 1, 3: 1})
        w1 = vec({0: 1, 3: 1})
        
        # m2 = (a10+a11)b00
        u2 = vec({2: 1, 3: 1})
        v2 = vec({0: 1})
        w2 = vec({2: 1, 3: -1})
        
        # m3 = a00(b10-b11)
        u3 = vec({0: 1})
        v3 = vec({2: 1, 3: -1})
        w3 = vec({1: 1, 3: 1})
        
        # m4 = a11(b01-b00)
        u4 = vec({3: 1})
        v4 = vec({1: 1, 0: -1})
        w4 = vec({0: 1, 2: 1})
        
        # m5 = (a00+a01)b11
        u5 = vec({0: 1, 1: 1})
        v5 = vec({3: 1})
        w5 = vec({0: -1, 1: 1})
        
        # m6 = (a10-a00)(b00+b01)
        u6 = vec({2: 1, 0: -1})
        v6 = vec({0: 1, 1: 1})
        w6 = vec({3: 1})
        
        # m7 = (a01-a11)(b10+b11)
        u7 = vec({1: 1, 3: -1})
        v7 = vec({2: 1, 3: 1})
        w7 = vec({0: 1})
        
        return [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7),
        ]
    
    coeffs = strassen_coeffs()
    
    # Define 4 blocks:
    # Block 0: A1 (cols 0-1) @ B11 (rows 0-1, cols 0-1) -> C cols 0-1
    # Block 1: A1 (cols 0-1) @ B12 (rows 0-1, cols 2-3) -> C cols 2-3
    # Block 2: A2 (cols 2-3) @ B21 (rows 2-3, cols 0-1) -> C cols 0-1
    # Block 3: A2 (cols 2-3) @ B22 (rows 2-3, cols 2-3) -> C cols 2-3
    
    # A indices: i in {0,1}, j in {0,1,2,3}
    # A local index for A1: (i, j) -> j in {0,1}
    # A local index for A2: (i, j) -> j in {2,3}, local col = j-2
    
    # B indices: j in {0,1,2,3}, k in {0,1,2,3}
    # B local index for B11: (j, k) -> j in {0,1}, k in {0,1}
    # B local index for B12: (j, k) -> j in {0,1}, k in {2,3}, local col = k-2
    # B local index for B21: (j, k) -> j in {2,3}, k in {0,1}, local row = j-2
    # B local index for B22: (j, k) -> j in {2,3}, k in {2,3}, local row = j-2, local col = k-2
    
    # C indices: i in {0,1}, k in {0,1,2,3}
    # C local index for cols 0-1: (i, k) -> k in {0,1}
    # C local index for cols 2-3: (i, k) -> k in {2,3}, local col = k-2
    
    for block in range(4):
        for s in range(7):
            col = block * 7 + s
            u, v, w = coeffs[s]
            
            # Determine which A block, which B block, which C block
            if block == 0:
                # A1, B11, C cols 0-1
                for i in range(2):
                    for j in range(2):
                        U[i * 4 + j][col] += u[j]
                for j in range(2):
                    for k in range(2):
                        V[j * 4 + k][col] += v[k]
                for i in range(2):
                    for k in range(2):
                        W[k * 2 + i][col] += w[k * 2 + i] if False else w[i * 2 + k]
                        # W index is k*n+i = k*2+i
                        # w local index is i*2+k
                        W[k * 2 + i][col] = w[i * 2 + k]
                        
            elif block == 1:
                # A1, B12, C cols 2-3
                for i in range(2):
                    for j in range(2):
                        U[i * 4 + j][col] += u[j]
                for j in range(2):
                    for k in range(2):
                        V[j * 4 + (k + 2)][col] += v[k]
                for i in range(2):
                    for k in range(2):
                        W[(k + 2) * 2 + i][col] = w[i * 2 + k]
                        
            elif block == 2:
                # A2, B21, C cols 0-1
                for i in range(2):
                    for j in range(2):
                        U[i * 4 + (j + 2)][col] += u[j]
                for j in range(2):
                    for k in range(2):
                        V[(j + 2) * 4 + k][col] += v[k]
                for i in range(2):
                    for k in range(2):
                        W[k * 2 + i][col] = w[i * 2 + k]
                        
            elif block == 3:
                # A2, B22, C cols 2-3
                for i in range(2):
                    for j in range(2):
                        U[i * 4 + (j + 2)][col] += u[j]
                for j in range(2):
                    for k in range(2):
                        V[(j + 2) * 4 + (k + 2)][col] += v[k]
                for i in range(2):
                    for k in range(2):
                        W[(k + 2) * 2 + i][col] = w[i * 2 + k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
