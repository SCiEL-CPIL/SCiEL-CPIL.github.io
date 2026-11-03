"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition:
    # Split m=5 into m1=3, m2=2
    # Split p=7 into p1=3, p2=4
    
    # Rank contributions:
    # 1. M(3,3,3): Rank 27 (Schoolbook)
    # 2. M(3,3,4): Rank 36 (Schoolbook)
    # 3. M(3,2,3): Rank 18 (Schoolbook)
    # 4. M(3,2,4): Rank 24 (Schoolbook)
    # Total Rank = 27 + 36 + 18 + 24 = 105
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Block 1: 3x3 by 3x3
    U1, V1, W1 = schoolbook(3, 3, 3)
    U_blocks.append(U1)
    V_blocks.append(V1)
    W_blocks.append(W1)
    
    # Block 2: 3x3 by 3x4
    U2, V2, W2 = schoolbook(3, 3, 4)
    U_blocks.append(U2)
    V_blocks.append(V2)
    W_blocks.append(W2)
    
    # Block 3: 3x2 by 2x3
    U3, V3, W3 = schoolbook(3, 2, 3)
    U_blocks.append(U3)
    V_blocks.append(V3)
    W_blocks.append(W3)
    
    # Block 4: 3x2 by 2x4
    U4, V4, W4 = schoolbook(3, 2, 4)
    U_blocks.append(U4)
    V_blocks.append(V4)
    W_blocks.append(W4)
    
    total_rank = sum(len(b[0]) for b in U_blocks)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Block 1
    for i in range(3):
        for j in range(3):
            u_row = i * m + j
            for c in range(len(U1[0])):
                U[u_row][col_offset + c] = U1[i * 3 + j][c]
    for j in range(3):
        for k in range(3):
            v_row = j * p + k
            for c in range(len(V1[0])):
                V[v_row][col_offset + c] = V1[j * 3 + k][c]
    for k in range(3):
        for i in range(3):
            w_row = k * n + i
            for c in range(len(W1[0])):
                W[w_row][col_offset + c] = W1[k * 3 + i][c]
    col_offset += len(U1[0])
    
    # Block 2
    for i in range(3):
        for j in range(3):
            u_row = i * m + j
            for c in range(len(U2[0])):
                U[u_row][col_offset + c] = U2[i * 3 + j][c]
    for j in range(3):
        for k in range(4):
            v_row = j * p + (k + 3)
            for c in range(len(V2[0])):
                V[v_row][col_offset + c] = V2[j * 4 + k][c]
    for k in range(4):
        for i in range(3):
            w_row = (k + 3) * n + i
            for c in range(len(W2[0])):
                W[w_row][col_offset + c] = W2[k * 3 + i][c]
    col_offset += len(U2[0])
    
    # Block 3
    for i in range(3):
        for j in range(2):
            u_row = i * m + (j + 3)
            for c in range(len(U3[0])):
                U[u_row][col_offset + c] = U3[i * 2 + j][c]
    for j in range(2):
        for k in range(3):
            v_row = (j + 3) * p + k
            for c in range(len(V3[0])):
                V[v_row][col_offset + c] = V3[j * 3 + k][c]
    for k in range(3):
        for i in range(3):
            w_row = k * n + i
            for c in range(len(W3[0])):
                W[w_row][col_offset + c] = W3[k * 3 + i][c]
    col_offset += len(U3[0])
    
    # Block 4
    for i in range(3):
        for j in range(2):
            u_row = i * m + (j + 3)
            for c in range(len(U4[0])):
                U[u_row][col_offset + c] = U4[i * 2 + j][c]
    for j in range(2):
        for k in range(4):
            v_row = (j + 3) * p + (k + 3)
            for c in range(len(V4[0])):
                V[v_row][col_offset + c] = V4[j * 4 + k][c]
    for k in range(4):
        for i in range(3):
            w_row = (k + 3) * n + i
            for c in range(len(W4[0])):
                W[w_row][col_offset + c] = W4[k * 3 + i][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
