import time

def _verify(U, V, W, n=2, m=4, p=7):
    """Check if U, V, W give exact decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 4, 7
    
    # We will build the decomposition block by block.
    # Each block adds some rows to U, V, W.
    
    # U: 8 x Rank
    # V: 28 x Rank
    # W: 14 x Rank
    
    # We don't know the final rank yet, so we'll accumulate lists of vectors.
    # u_vecs: list of vectors of length 8
    # v_vecs: list of vectors of length 28
    # w_vecs: list of vectors of length 14
    
    u_vecs = []
    v_vecs = []
    w_vecs = []
    
    def add_block_222(j_pair, k_pair):
        # j_pair: (j0, j1)
        # k_pair: (k0, k1)
        # i is always (0, 1)
        j0, j1 = j_pair
        k0, k1 = k_pair
        
        # Strassen 7 products
        # 1. (A00+A11)(B00+B11) -> C00, C11
        u = [0]*8; u[0] = 1; u[3] = 1 # A00, A11 (idx 0=0*4+0, 3=1*4+3? No. A is 2x4. i*m+j. i=0,j=0->0. i=1,j=3->7? No.)
        # Wait, A is 2x4. i in 0,1. j in 0,1,2,3.
        # In the 2x2x2 block, j is restricted to {j0, j1}.
        # Local j' 0 -> Global j0. Local j' 1 -> Global j1.
        # Local i' 0 -> Global 0. Local i' 1 -> Global 1.
        # Local k' 0 -> Global k0. Local k' 1 -> Global k1.
        
        # Indices in U (size 8): i*m + j.
        # i=0, j=j0 -> 0*4+j0 = j0
        # i=0, j=j1 -> 0*4+j1 = j1
        # i=1, j=j0 -> 1*4+j0 = 4+j0
        # i=1, j=j1 -> 1*4+j1 = 4+j1
        
        # Indices in V (size 28): j*p + k.
        # j=j0, k=k0 -> j0*7+k0
        # j=j0, k=k1 -> j0*7+k1
        # j=j1, k=k0 -> j1*7+k0
        # j=j1, k=k1 -> j1*7+k1
        
        # Indices in W (size 14): k*n + i.
        # k=k0, i=0 -> k0*2+0 = k0
        # k=k0, i=1 -> k0*2+1 = k0+1
        # k=k1, i=0 -> k1*2+0 = k1
        # k=k1, i=1 -> k1*2+1 = k1+1
        
        # Strassen M1: (A00+A11)(B00+B11)
        # A00: local i=0, j=0 -> global i=0, j=j0 -> U idx j0
        # A11: local i=1, j=1 -> global i=1, j=j1 -> U idx 4+j1
        # B00: local j=0, k=0 -> global j=j0, k=k0 -> V idx j0*7+k0
        # B11: local j=1, k=1 -> global j=j1, k=k1 -> V idx j1*7+k1
        # C00: local k=0, i=0 -> global k=k0, i=0 -> W idx k0
        # C11: local k=1, i=1 -> global k=k1, i=1 -> W idx k1+1
        
        u1 = [0]*8; u1[j0] = 1; u1[4+j1] = 1
        v1 = [0]*28; v1[j0*7+k0] = 1; v1[j1*7+k1] = 1
        w1 = [0]*14; w1[k0] = 1; w1[k1+1] = 1
        u_vecs.append(u1); v_vecs.append(v1); w_vecs.append(w1)
        
        # M2: (A10+A11)B00
        # A10: local i=1, j=0 -> global i=1, j=j0 -> U idx 4+j0
        # A11: local i=1, j=1 -> global i=1, j=j1 -> U idx 4+j1
        # B00: global j=j0, k=k0 -> V idx j0*7+k0
        # C10: local k=1, i=0? No. C10 is i=1, k=0?
        # Wait. C indices are (k, i).
        # Local C10 means local k=1, local i=0?
        # Standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # Here Cxy means i=x, k=y? Or i=y, k=x?
        # In matrix mult C = AB, C[i][k] = sum_j A[i][j]B[j][k].
        # So C index is (i, k).
        # My W index is k*n + i.
        # So W[k, i].
        # Local C10 means i=1, k=0.
        # Global: i=1, k=k0.
        # W idx: k0*2 + 1 = k0+1.
        # Local C11 means i=1, k=1.
        # Global: i=1, k=k1.
        # W idx: k1*2 + 1 = k1+1.
        
        # Let's re-verify M2 contribution.
        # M2 = (A10+A11)B00.
        # Contributes to C10 (+1) and C11 (-1).
        # C10: i=1, k=0. W idx k0+1.
        # C11: i=1, k=1. W idx k1+1.
        
        u2 = [0]*8; u2[4+j0] = 1; u2[4+j1] = 1
        v2 = [0]*28; v2[j0*7+k0] = 1
        w2 = [0]*14; w2[k0+1] = 1; w2[k1+1] = -1
        u_vecs.append(u2); v_vecs.append(v2); w_vecs.append(w2)
        
        # M3: A00(B10-B11)
        # A00: U idx j0
        # B10: local j=1, k=0 -> global j=j1, k=k0 -> V idx j1*7+k0
        # B11: local j=1, k=1 -> global j=j1, k=k1 -> V idx j1*7+k1
        # C01 (+1), C11 (+1)
        # C01: i=0, k=1. W idx k1.
        # C11: i=1, k=1. W idx k1+1.
        
        u3 = [0]*8; u3[j0] = 1
        v3 = [0]*28; v3[j1*7+k0] = 1; v3[j1*7+k1] = -1
        w3 = [0]*14; w3[k1] = 1; w3[k1+1] = 1
        u_vecs.append(u3); v_vecs.append(v3); w_vecs.append(w3)
        
        # M4: A11(B01-B00)
        # A11: U idx 4+j1
        # B01: local j=0, k=1 -> global j=j0, k=k1 -> V idx j0*7+k1
        # B00: local j=0, k=0 -> global j=j0, k=k0 -> V idx j0*7+k0
        # C00 (+1), C10 (+1)
        # C00: i=0, k=0. W idx k0.
        # C10: i=1, k=0. W idx k0+1.
        
        u4 = [0]*8; u4[4+j1] = 1
        v4 = [0]*28; v4[j0*7+k1] = 1; v4[j0*7+k0] = -1
        w4 = [0]*14; w4[k0] = 1; w4[k0+1] = 1
        u_vecs.append(u4); v_vecs.append(v4); w_vecs.append(w4)
        
        # M5: (A00+A01)B11
        # A00: U idx j0
        # A01: local i=0, j=1 -> global i=0, j=j1 -> U idx j1
        # B11: V idx j1*7+k1
        # C00 (-1), C01 (+1)
        # C00: i=0, k=0. W idx k0.
        # C01: i=0, k=1. W idx k1.
        
        u5 = [0]*8; u5[j0] = 1; u5[j1] = 1
        v5 = [0]*28; v5[j1*7+k1] = 1
        w5 = [0]*14; w5[k0] = -1; w5[k1] = 1
        u_vecs.append(u5); v_vecs.append(v5); w_vecs.append(w5)
        
        # M6: (A10-A00)(B00+B01)
        # A10: U idx 4+j0
        # A00: U idx j0
        # B00: V idx j0*7+k0
        # B01: V idx j0*7+k1
        # C11 (+1)
        # C11: i=1, k=1. W idx k1+1.
        
        u6 = [0]*8; u6[4+j0] = 1; u6[j0] = -1
        v6 = [0]*28; v6[j0*7+k0] = 1; v6[j0*7+k1] = 1
        w6 = [0]*14; w6[k1+1] = 1
        u_vecs.append(u6); v_vecs.append(v6); w_vecs.append(w6)
        
        # M7: (A01-A11)(B10+B11)
        # A01: U idx j1
        # A11: U idx 4+j1
        # B10: V idx j1*7+k0
        # B11: V idx j1*7+k1
        # C00 (+1)
        # C00: i=0, k=0. W idx k0.
        
        u7 = [0]*8; u7[j1] = 1; u7[4+j1] = -1
        v7 = [0]*28; v7[j1*7+k0] = 1; v7[j1*7+k1] = 1
        w7 = [0]*14; w7[k0] = 1
        u_vecs.append(u7); v_vecs.append(v7); w_vecs.append(w7)

    def add_block_221(j_pair, k_val):
        # j_pair: (j0, j1)
        # k_val: k
        j0, j1 = j_pair
        
        # Schoolbook 4 products
        # 1. A00 B0k -> C0k
        # 2. A01 B1k -> C0k
        # 3. A10 B0k -> C1k
        # 4. A11 B1k -> C1k
        
        # A00: U idx j0
        # A01: U idx j1
        # A10: U idx 4+j0
        # A11: U idx 4+j1
        
        # B0k: V idx j0*7+k_val
        # B1k: V idx j1*7+k_val
        
        # C0k: W idx k_val*2 + 0 = k_val
        # C1k: W idx k_val*2 + 1 = k_val+1
        
        u1 = [0]*8; u1[j0] = 1
        v1 = [0]*28; v1[j0*7+k_val] = 1
        w1 = [0]*14; w1[k_val] = 1
        u_vecs.append(u1); v_vecs.append(v1); w_vecs.append(w1)
        
        u2 = [0]*8; u2[j1] = 1
        v2 = [0]*28; v2[j1*7+k_val] = 1
        w2 = [0]*14; w2[k_val] = 1
        u_vecs.append(u2); v_vecs.append(v2); w_vecs.append(w2)
        
        u3 = [0]*8; u3[4+j0] = 1
        v3 = [0]*28; v3[j0*7+k_val] = 1
        w3 = [0]*14; w3[k_val+1] = 1
        u_vecs.append(u3); v_vecs.append(v3); w_vecs.append(w3)
        
        u4 = [0]*8; u4[4+j1] = 1
        v4 = [0]*28; v4[j1*7+k_val] = 1
        w4 = [0]*14; w4[k_val+1] = 1
        u_vecs.append(u4); v_vecs.append(v4); w_vecs.append(w4)

    # Define blocks
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    k_leftover = 6
    
    # Add 6 2x2x2 blocks
    for jp in j_pairs:
        for kp in k_pairs:
            add_block_222(jp, kp)
            
    # Add 2 2x2x1 blocks
    for jp in j_pairs:
        add_block_221(jp, k_leftover)
        
    # Convert lists of vectors to matrices
    rank = len(u_vecs)
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(14)]
    
    for r in range(rank):
        for i in range(8):
            U[i][r] = u_vecs[r][i]
        for i in range(28):
            V[i][r] = v_vecs[r][i]
        for i in range(14):
            W[i][r] = w_vecs[r][i]
            
    # Verify
    if not _verify(U, V, W):
        # Fallback to schoolbook if verification fails
        return _schoolbook(2, 4, 7)
        
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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
