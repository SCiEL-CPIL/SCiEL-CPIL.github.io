import numpy as np

def _strassen_222():
    """Returns U, V, W for 2x2x2 Strassen decomposition (Rank 7)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    # 0-indexed: i,j,k in {0,1}
    
    # Strassen M1..M7
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We define U, V, W for each M
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1 # A11, A22
    V[0, 0] = 1; V[3, 0] = 1 # B11, B22
    W[0, 0] = 1; W[3, 0] = 1 # C11, C22
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1 # A21, A22
    V[0, 1] = 1 # B11
    W[2, 1] = 1; W[3, 1] = 1 # C21, C22
    
    # M3
    U[0, 2] = 1 # A11
    V[2, 2] = 1; V[3, 2] = -1 # B21, B22
    W[1, 2] = 1; W[3, 2] = 1 # C12, C22
    
    # M4
    U[3, 3] = 1 # A22
    V[0, 3] = 1; V[2, 3] = -1 # B11, B21
    W[0, 3] = 1; W[2, 3] = 1 # C11, C21
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1 # A11, A12
    V[3, 4] = 1 # B22
    W[0, 4] = -1; W[1, 4] = 1 # C11, C12
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1 # A21, A11
    V[0, 5] = 1; V[1, 5] = 1 # B11, B12
    W[3, 5] = 1 # C22
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1 # A12, A22
    V[2, 6] = 1; V[3, 6] = 1 # B21, B22
    W[0, 6] = 1 # C11
    
    return U, V, W

def _schoolbook_221():
    """Returns U, V, W for 2x2x1 schoolbook (Rank 4)."""
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1
            V[j*p, col] = 1 # k=0
            W[0*n+i, col] = 1 # k=0
            col += 1
    return U, V, W

def _combine_blocks(blocks):
    """Combines a list of (U, V, W) blocks horizontally."""
    if not blocks:
        return np.zeros((0,0)), np.zeros((0,0)), np.zeros((0,0))
    U = np.hstack([b[0] for b in blocks])
    V = np.hstack([b[1] for b in blocks])
    W = np.hstack([b[2] for b in blocks])
    return U, V, W

def construct():
    # Target: 3x4x5
    n, m, p = 3, 4, 5
    
    # Split n=3 into 2+1
    # Part 1: 2x4x5
    # Part 2: 1x4x5
    
    # --- Part 2: 1x4x5 (Schoolbook Rank 4) ---
    n2, m2, p2 = 1, 4, 5
    rank2 = n2 * m2 * p2
    U2 = np.zeros((n2*m2, rank2))
    V2 = np.zeros((m2*p2, rank2))
    W2 = np.zeros((p2*n2, rank2))
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i*m2+j, col] = 1
                V2[j*p2+k, col] = 1
                W2[k*n2+i, col] = 1
                col += 1
                
    # --- Part 1: 2x4x5 ---
    # Split m=4 into 2+2
    # Block 1.1: 2x2x5
    # Block 1.2: 2x2x5
    
    # --- Block 2x2x5 ---
    # Split p=5 into 2+3
    # Sub 2.1: 2x2x2 (Strassen)
    # Sub 2.2: 2x2x3
    
    # --- Sub 2x2x3 ---
    # Split p=3 into 2+1
    # Sub 2.2.1: 2x2x2 (Strassen)
    # Sub 2.2.2: 2x2x1 (Schoolbook)
    
    S_222_1, V_222_1, W_222_1 = _strassen_222()
    S_222_2, V_222_2, W_222_2 = _strassen_222()
    S_221, V_221, W_221 = _schoolbook_221()
    
    # Combine 2x2x3
    U_223, V_223, W_223 = _combine_blocks([(S_222_1, V_222_1, W_222_1), (S_221, V_221, W_221)])
    
    # Combine 2x2x5
    S_222_3, V_222_3, W_222_3 = _strassen_222()
    U_225, V_225, W_225 = _combine_blocks([(S_222_3, V_222_3, W_222_3), (U_223, V_223, W_223)])
    
    # Combine 2x4x5 (two 2x2x5 blocks)
    U_245, V_245, W_245 = _combine_blocks([(U_225, V_225, W_225), (U_225, V_225, W_225)])
    
    # --- Final Assembly ---
    # We need to map the local indices to global indices.
    # This is tricky with numpy hstack.
    # I will construct the final matrices manually or use a helper to offset indices.
    
    # Helper to offset a block
    def offset_block(U, V, W, i_off, j_off, k_off, n, m, p):
        # U is (n*m, r), V is (m*p, r), W is (p*n, r)
        # We need to place these into global U, V, W.
        # Global U index: i*m + j. Local: i'*m + j'.
        # Global V index: j*p + k. Local: j'*p + k'.
        # Global W index: k*n + i. Local: k'*n + i'.
        
        r = U.shape[1]
        gU = np.zeros((n*m, r))
        gV = np.zeros((m*p, r))
        gW = np.zeros((p*n, r))
        
        for c in range(r):
            # Map U
            for idx in range(U.shape[0]):
                if U[idx, c] != 0:
                    li, lj = divmod(idx, m)
                    gi = li + i_off
                    gj = lj + j_off
                    if gi < n and gj < m:
                        gU[gi*m + gj, c] = U[idx, c]
                        
            # Map V
            for idx in range(V.shape[0]):
                if V[idx, c] != 0:
                    lj, lk = divmod(idx, p)
                    gj = lj + j_off
                    gk = lk + k_off
                    if gj < m and gk < p:
                        gV[gj*p + gk, c] = V[idx, c]
                        
            # Map W
            for idx in range(W.shape[0]):
                if W[idx, c] != 0:
                    lk, li = divmod(idx, n)
                    gk = lk + k_off
                    gi = li + i_off
                    if gk < p and gi < n:
                        gW[gk*n + gi, c] = W[idx, c]
                        
        return gU, gV, gW

    # We need to combine the blocks with correct offsets.
    # Part 2: 1x4x5 at i=2, j=0, k=0
    gU2, gV2, gW2 = offset_block(U2, V2, W2, 2, 0, 0, n, m, p)
    
    # Part 1.1: 2x2x5 at i=0, j=0, k=0
    gU11, gV11, gW11 = offset_block(U_225, V_225, W_225, 0, 0, 0, n, m, p)
    
    # Part 1.2: 2x2x5 at i=0, j=2, k=0
    gU12, gV12, gW12 = offset_block(U_225, V_225, W_225, 0, 2, 0, n, m, p)
    
    # Combine all
    U_final = np.hstack([gU11, gU12, gU2])
    V_final = np.hstack([gV11, gV12, gV2])
    W_final = np.hstack([gW11, gW12, gW2])
    
    # Convert to lists for the required format
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
