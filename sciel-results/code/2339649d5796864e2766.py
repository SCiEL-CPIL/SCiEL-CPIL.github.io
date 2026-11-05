"""Matrix multiplication 3x5x7 with block decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=5, p=7
    
    # Block decomposition: n=2+1, m=2+3, p=3+4
    n1, n2 = 2, 1
    m1, m2 = 2, 3
    p1, p2 = 3, 4
    
    # We'll build lists of (U_block, V_block, W_block, n_off, m_off, p_off)
    # where n_off, m_off, p_off are the offsets in the global matrices
    
    blocks = []
    
    def trivial(n_b, m_b, p_b):
        """Return (U, V, W) for trivial decomposition of n_b x m_b x p_b."""
        rank = n_b * m_b * p_b
        U = [[0]*rank for _ in range(n_b * m_b)]
        V = [[0]*rank for _ in range(m_b * p_b)]
        W = [[0]*rank for _ in range(p_b * n_b)]
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i*m_b + j][col] = 1
                    V[j*p_b + k][col] = 1
                    W[k*n_b + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    def strassen():
        """Return (U, V, W) for Strassen 2x2x2 decomposition, rank 7."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # M1 = (a+d)(e+h)
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1; W[3][0] += 1
        
        # M2 = (c+d)e
        U[2][1] += 1; U[3][1] += 1
        V[0][1] += 1
        W[2][1] += 1; W[3][1] -= 1
        
        # M3 = a(f-h)
        U[0][2] += 1
        V[1][2] += 1; V[3][2] -= 1
        W[1][2] += 1; W[3][2] += 1
        
        # M4 = d(g-f)
        U[3][3] += 1
        V[2][3] += 1; V[1][3] -= 1
        W[0][3] += 1; W[2][3] += 1
        
        # M5 = (a+b)h
        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        W[0][4] -= 1; W[1][4] += 1
        
        # M6 = (c-a)(e+f)
        U[2][5] += 1; U[0][5] -= 1
        V[0][5] += 1; V[1][5] += 1
        W[3][5] += 1
        
        # M7 = (b-d)(g+h)
        U[1][6] += 1; U[3][6] -= 1
        V[2][6] += 1; V[3][6] += 1
        W[0][6] += 1
        
        return U, V, W, rank
    
    # 8 sub-multiplications from block decomposition:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # 1. A11(n1 x m1) * B11(m1 x p1) -> (n1 x p1)
    #    n1=2, m1=2, p1=3. Split p1=2+1.
    #    (2x2)*(2x2) + (2x2)*(2x1)
    U1a, V1a, W1a, r1a = strassen()
    blocks.append((U1a, V1a, W1a, r1a, 0, 0, 0, 2, 2, 2))  # n_off, m_off, p_off, n_b, m_b, p_b
    
    U1b, V1b, W1b, r1b = trivial(2, 2, 1)
    blocks.append((U1b, V1b, W1b, r1b, 0, 0, 2, 2, 2, 1))
    
    # 2. A12(n1 x m2) * B21(m2 x p1) -> (n1 x p1)
    #    n1=2, m2=3, p1=3. Split m2=2+1, p1=2+1.
    #    (2x2)*(2x2) + (2x2)*(2x1) + (2x1)*(1x2) + (2x1)*(1x1)
    U2a, V2a, W2a, r2a = strassen()
    blocks.append((U2a, V2a, W2a, r2a, 0, 2, 0, 2, 2, 2))
    
    U2b, V2b, W2b, r2b = trivial(2, 2, 1)
    blocks.append((U2b, V2b, W2b, r2b, 0, 2, 2, 2, 2, 1))
    
    U2c, V2c, W2c, r2c = trivial(2, 1, 2)
    blocks.append((U2c, V2c, W2c, r2c, 0, 4, 0, 2, 1, 2))
    
    U2d, V2d, W2d, r2d = trivial(2, 1, 1)
    blocks.append((U2d, V2d, W2d, r2d, 0, 4, 2, 2, 1, 1))
    
    # 3. A11(n1 x m1) * B12(m1 x p2) -> (n1 x p2)
    #    n1=2, m1=2, p2=4. Split p2=2+2.
    #    (2x2)*(2x2) + (2x2)*(2x2)
    U3a, V3a, W3a, r3a = strassen()
    blocks.append((U3a, V3a, W3a, r3a, 0, 0, 3, 2, 2, 2))
    
    U3b, V3b, W3b, r3b = strassen()
    blocks.append((U3b, V3b, W3b, r3b, 0, 0, 5, 2, 2, 2))
    
    # 4. A12(n1 x m2) * B22(m2 x p2) -> (n1 x p2)
    #    n1=2, m2=3, p2=4. Split m2=2+1, p2=2+2.
    #    (2x2)*(2x2) + (2x2)*(2x2) + (2x1)*(1x2) + (2x1)*(1x2)
    U4a, V4a, W4a, r4a = strassen()
    blocks.append((U4a, V4a, W4a, r4a, 0, 2, 3, 2, 2, 2))
    
    U4b, V4b, W4b, r4b = strassen()
    blocks.append((U4b, V4b, W4b, r4b, 0, 2, 5, 2, 2, 2))
    
    U4c, V4c, W4c, r4c = trivial(2, 1, 2)
    blocks.append((U4c, V4c, W4c, r4c, 0, 4, 3, 2, 1, 2))
    
    U4d, V4d, W4d, r4d = trivial(2, 1, 2)
    blocks.append((U4d, V4d, W4d, r4d, 0, 4, 5, 2, 1, 2))
    
    # 5. A21(n2 x m1) * B11(m1 x p1) -> (n2 x p1)
    #    n2=1, m1=2, p1=3. Split p1=2+1.
    #    (1x2)*(2x2) + (1x2)*(2x1)
    U5a, V5a, W5a, r5a = trivial(1, 2, 2)
    blocks.append((U5a, V5a, W5a, r5a, 2, 0, 0, 1, 2, 2))
    
    U5b, V5b, W5b, r5b = trivial(1, 2, 1)
    blocks.append((U5b, V5b, W5b, r5b, 2, 0, 2, 1, 2, 1))
    
    # 6. A22(n2 x m2) * B21(m2 x p1) -> (n2 x p1)
    #    n2=1, m2=3, p1=3. Split m2=2+1, p1=2+1.
    #    (1x2)*(2x2) + (1x2)*(2x1) + (1x1)*(1x2) + (1x1)*(1x1)
    U6a, V6a, W6a, r6a = trivial(1, 2, 2)
    blocks.append((U6a, V6a, W6a, r6a, 2, 2, 0, 1, 2, 2))
    
    U6b, V6b, W6b, r6b = trivial(1, 2, 1)
    blocks.append((U6b, V6b, W6b, r6b, 2, 2, 2, 1, 2, 1))
    
    U6c, V6c, W6c, r6c = trivial(1, 1, 2)
    blocks.append((U6c, V6c, W6c, r6c, 2, 4, 0, 1, 1, 2))
    
    U6d, V6d, W6d, r6d = trivial(1, 1, 1)
    blocks.append((U6d, V6d, W6d, r6d, 2, 4, 2, 1, 1, 1))
    
    # 7. A21(n2 x m1) * B12(m1 x p2) -> (n2 x p2)
    #    n2=1, m1=2, p2=4. Split p2=2+2.
    #    (1x2)*(2x2) + (1x2)*(2x2)
    U7a, V7a, W7a, r7a = trivial(1, 2, 2)
    blocks.append((U7a, V7a, W7a, r7a, 2, 0, 3, 1, 2, 2))
    
    U7b, V7b, W7b, r7b = trivial(1, 2, 2)
    blocks.append((U7b, V7b, W7b, r7b, 2, 0, 5, 1, 2, 2))
    
    # 8. A22(n2 x m2) * B22(m2 x p2) -> (n2 x p2)
    #    n2=1, m2=3, p2=4. Split m2=2+1, p2=2+2.
    #    (1x2)*(2x2) + (1x2)*(2x2) + (1x1)*(1x2) + (1x1)*(1x2)
    U8a, V8a, W8a, r8a = trivial(1, 2, 2)
    blocks.append((U8a, V8a, W8a, r8a, 2, 2, 3, 1, 2, 2))
    
    U8b, V8b, W8b, r8b = trivial(1, 2, 2)
    blocks.append((U8b, V8b, W8b, r8b, 2, 2, 5, 1, 2, 2))
    
    U8c, V8c, W8c, r8c = trivial(1, 1, 2)
    blocks.append((U8c, V8c, W8c, r8c, 2, 4, 3, 1, 1, 2))
    
    U8d, V8d, W8d, r8d = trivial(1, 1, 2)
    blocks.append((U8d, V8d, W8d, r8d, 2, 4, 5, 1, 1, 2))
    
    # Assemble global matrices
    total_rank = sum(b[3] for b in blocks)
    U_full = [[0]*total_rank for _ in range(n*m)]  # 15 x total_rank
    V_full = [[0]*total_rank for _ in range(m*p)]  # 35 x total_rank
    W_full = [[0]*total_rank for _ in range(p*n)]  # 21 x total_rank
    
    col_offset = 0
    for U_b, V_b, W_b, rank_b, n_off, m_off, p_off, n_b, m_b, p_b in blocks:
        # U_b: (n_b*m_b) x rank_b, local index (i*m_b + j, r)
        # Global U index: ((n_off + i)*m + (m_off + j), col_offset + r)
        for i in range(n_b):
            for j in range(m_b):
                for r in range(rank_b):
                    val = U_b[i*m_b + j][r]
                    if val != 0:
                        U_full[(n_off + i)*m + (m_off + j)][col_offset + r] += val
        
        # V_b: (m_b*p_b) x rank_b, local index (j*p_b + k, r)
        # Global V index: ((m_off + j)*p + (p_off + k), col_offset + r)
        for j in range(m_b):
            for k in range(p_b):
                for r in range(rank_b):
                    val = V_b[j*p_b + k][r]
                    if val != 0:
                        V_full[(m_off + j)*p + (p_off + k)][col_offset + r] += val
        
        # W_b: (p_b*n_b) x rank_b, local index (k*n_b + i, r)
        # Global W index: ((p_off + k)*n + (n_off + i), col_offset + r)
        for k in range(p_b):
            for i in range(n_b):
                for r in range(rank_b):
                    val = W_b[k*n_b + i][r]
                    if val != 0:
                        W_full[(p_off + k)*n + (n_off + i)][col_offset + r] += val
        
        col_offset += rank_b
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
