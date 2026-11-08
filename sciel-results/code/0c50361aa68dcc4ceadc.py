import numpy as np

def construct():
    n, m, p = 2, 5, 6
    
    # We will build the decomposition by summing block decompositions.
    # Each block is defined by (n_block, m_block, p_block) and its factor matrices.
    # We accumulate U, V, W.
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We don't know the total rank yet, so we'll collect lists of factors
    # and then concatenate them.
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Helper to add a schoolbook block
    def add_schoolbook(n_b, m_b, p_b, i_off, j_off, k_off):
        rank_b = n_b * m_b * p_b
        U_b = [[0] * rank_b for _ in range(n_b * m_b)]
        V_b = [[0] * rank_b for _ in range(m_b * p_b)]
        W_b = [[0] * rank_b for _ in range(p_b * n_b)]
        
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U_b[i * m_b + j][col] = 1
                    V_b[j * p_b + k][col] = 1
                    W_b[k * n_b + i][col] = 1
                    col += 1
        
        # Embed into global coordinates
        # U index: (i_off + i) * m + (j_off + j)
        # V index: (j_off + j) * p + (k_off + k)
        # W index: (k_off + k) * n + (i_off + i)
        
        # We need to expand U_b, V_b, W_b to full sizes (n*m, rank_b), etc.
        U_full = [[0] * rank_b for _ in range(n * m)]
        V_full = [[0] * rank_b for _ in range(m * p)]
        W_full = [[0] * rank_b for _ in range(p * n)]
        
        for i in range(n_b):
            for j in range(m_b):
                u_row = (i_off + i) * m + (j_off + j)
                for c in range(rank_b):
                    U_full[u_row][c] = U_b[i * m_b + j][c]
                    
        for j in range(m_b):
            for k in range(p_b):
                v_row = (j_off + j) * p + (k_off + k)
                for c in range(rank_b):
                    V_full[v_row][c] = V_b[j * p_b + k][c]
                    
        for k in range(p_b):
            for i in range(n_b):
                w_row = (k_off + k) * n + (i_off + i)
                for c in range(rank_b):
                    W_full[w_row][c] = W_b[k * n_b + i][c]
                    
        U_blocks.append(U_full)
        V_blocks.append(V_full)
        W_blocks.append(W_full)

    # Helper to add Strassen block (2x2x2)
    def add_strassen(i_off, j_off, k_off):
        # Strassen rank 7
        # U, V, W for 2x2x2
        # Indices: i,j,k in {0,1}
        # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        
        # Strassen factors:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # U vectors for A (2x2)
        # A is indexed by i (row), j (col).
        # U[c] is a vector of length 4 (flattened A).
        
        U_b = [[0] * 7 for _ in range(4)]
        V_b = [[0] * 7 for _ in range(4)]
        W_b = [[0] * 7 for _ in range(4)]
        
        # Fill U_b (A)
        # M1: A11(1) + A22(1) -> U[0,0]=1, U[0,3]=1 (if 0:11, 1:12, 2:21, 3:22)
        # Wait, flattening order: i*m+j. m=2.
        # 0: 0,0. 1: 0,1. 2: 1,0. 3: 1,1.
        
        # M1: A00 + A11
        U_b[0][0] = 1; U_b[0][3] = 1
        # M2: A10 + A11
        U_b[1][2] = 1; U_b[1][3] = 1
        # M3: A00
        U_b[2][0] = 1
        # M4: A11
        U_b[3][3] = 1
        # M5: A00 + A01
        U_b[4][0] = 1; U_b[4][1] = 1
        # M6: A10 - A00
        U_b[5][2] = 1; U_b[5][0] = -1
        # M7: A01 - A11
        U_b[6][1] = 1; U_b[6][3] = -1
        
        # Fill V_b (B)
        # B is indexed by j (row), k (col).
        # 0: 0,0. 1: 0,1. 2: 1,0. 3: 1,1.
        
        # M1: B00 + B11
        V_b[0][0] = 1; V_b[0][3] = 1
        # M2: B00
        V_b[1][0] = 1
        # M3: B01 - B11
        V_b[2][1] = 1; V_b[2][3] = -1
        # M4: B10 - B00
        V_b[3][2] = 1; V_b[3][0] = -1
        # M5: B11
        V_b[4][3] = 1
        # M6: B00 + B01
        V_b[5][0] = 1; V_b[5][1] = 1
        # M7: B10 + B11
        V_b[6][2] = 1; V_b[6][3] = 1
        
        # Fill W_b (C)
        # C is indexed by k (col of B), i (row of A).
        # W[k*2+i, c].
        # 0: k=0,i=0 (C00). 1: k=0,i=1 (C10). 2: k=1,i=0 (C01). 3: k=1,i=1 (C11).
        
        # C00 = M1 + M4 - M5 - M7
        W_b[0][0] = 1; W_b[0][3] = 1; W_b[0][4] = -1; W_b[0][6] = -1
        # C10 = M2 + M7
        W_b[1][1] = 1; W_b[1][6] = 1
        # C01 = M3 + M5
        W_b[2][2] = 1; W_b[2][4] = 1
        # C11 = M1 - M2 + M3 + M6
        W_b[3][0] = 1; W_b[3][1] = -1; W_b[3][2] = 1; W_b[3][5] = 1
        
        # Embed into global coordinates
        rank_b = 7
        U_full = [[0] * rank_b for _ in range(n * m)]
        V_full = [[0] * rank_b for _ in range(m * p)]
        W_full = [[0] * rank_b for _ in range(p * n)]
        
        # U: i in 0..1, j in 0..1
        for i in range(2):
            for j in range(2):
                u_row = (i_off + i) * m + (j_off + j)
                for c in range(7):
                    U_full[u_row][c] = U_b[i * 2 + j][c]
                    
        # V: j in 0..1, k in 0..1
        for j in range(2):
            for k in range(2):
                v_row = (j_off + j) * p + (k_off + k)
                for c in range(7):
                    V_full[v_row][c] = V_b[j * 2 + k][c]
                    
        # W: k in 0..1, i in 0..1
        for k in range(2):
            for i in range(2):
                w_row = (k_off + k) * n + (i_off + i)
                for c in range(7):
                    W_full[w_row][c] = W_b[k * 2 + i][c]
                    
        U_blocks.append(U_full)
        V_blocks.append(V_full)
        W_blocks.append(W_full)

    # Block 1: 2x4x4 at (i=0, j=0, k=0)
    # Split into 4 Strassen blocks:
    # (i=0, j=0, k=0), (i=0, j=0, k=2), (i=0, j=2, k=0), (i=0, j=2, k=2)
    # Wait, j is 0..3, k is 0..3.
    # Strassen covers j=0,1 and k=0,1.
    # Block 1a: j=0..1, k=0..1 -> Strassen at (0,0,0)
    # Block 1b: j=0..1, k=2..3 -> Strassen at (0,0,2)
    # Block 1c: j=2..3, k=0..1 -> Strassen at (0,2,0)
    # Block 1d: j=2..3, k=2..3 -> Strassen at (0,2,2)
    
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # Block 2: 2x4x1 at (i=0, j=0, k=4)
    # p=1, so k=4 only. m=4, so j=0..3.
    add_schoolbook(2, 4, 1, 0, 0, 4)
    
    # Block 3: 2x1x5 at (i=0, j=4, k=0)
    # m=1, so j=4 only. p=5, so k=0..4.
    add_schoolbook(2, 1, 5, 0, 4, 0)
    
    # Block 4: 2x5x1 at (i=0, j=0, k=5)
    # p=1, so k=5 only. m=5, so j=0..4.
    add_schoolbook(2, 5, 1, 0, 0, 5)

    # Concatenate
    total_rank = sum(len(u[0]) for u in U_blocks)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    for u_b, v_b, w_b in zip(U_blocks, V_blocks, W_blocks):
        rank_b = len(u_b[0])
        for r in range(n * m):
            for c in range(rank_b):
                U[r][offset + c] = u_b[r][c]
        for r in range(m * p):
            for c in range(rank_b):
                V[r][offset + c] = v_b[r][c]
        for r in range(p * n):
            for c in range(rank_b):
                W[r][offset + c] = w_b[r][c]
        offset += rank_b
        
    return {"U": U, "V": V, "W": W}

