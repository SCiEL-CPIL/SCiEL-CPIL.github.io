import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition indices
    # A is 2x5, B is 5x6
    # Split m=5 into [2, 2, 1]
    # Split p=6 into [2, 2, 2]
    
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    # Calculate offsets
    m_offsets = [0]
    for s in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + s)
        
    p_offsets = [0]
    for s in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + s)
        
    U = []
    V = []
    W = []
    
    # Helper to add a block decomposition
    def add_block(i_start, i_end, j_start, j_end, k_start, k_end, use_strassen):
        # i is 0..n-1 (always 0..1)
        # j is j_start..j_end-1
        # k is k_start..k_end-1
        ni = i_end - i_start
        nj = j_end - j_start
        nk = k_end - k_start
        
        if use_strassen and ni == 2 and nj == 2 and nk == 2:
            # Strassen 2x2x2
            # Local indices 0,1 for i, j, k
            # Map to global
            # M1: (A11+A22)(B11+B22)
            # U: (i,j) -> 1 if (i,j) in {(0,0),(1,1)} for A part? No.
            # Strassen terms are linear combinations.
            # We need to add 7 rank-1 tensors.
            
            # Term 1: (A11+A22) * (B11+B22)
            # U: 1 at (0,0), (1,1)
            # V: 1 at (0,0), (1,1)
            # W: 1 at (0,0), (1,1) -> C11, C22 parts
            
            # This is complex to map directly to U,V,W.
            # It is easier to generate the 7 products and their contributions to C.
            
            # Product 1: M1 = (A00+A11)(B00+B11)
            # U: 1 at (0,0), (1,1)
            # V: 1 at (0,0), (1,1)
            # W: contributes to C00, C11
            # C00 += M1, C11 += M1
            
            # We will append to U, V, W lists.
            
            # Term 1
            u = [0]*ni*nj
            v = [0]*nj*nk
            w = [0]*nk*ni
            u[0*ni+0] = 1 # A00
            u[1*ni+1] = 1 # A11
            v[0*nk+0] = 1 # B00
            v[1*nk+1] = 1 # B11
            # W: C00 += 1, C11 += 1
            w[0*ni+0] = 1 # C00
            w[1*ni+1] = 1 # C11
            U.append(u); V.append(v); W.append(w)
            
            # Term 2: M2 = (A10+A11)B00
            u = [0]*ni*nj
            v = [0]*nj*nk
            w = [0]*nk*ni
            u[1*ni+0] = 1 # A10
            u[1*ni+1] = 1 # A11
            v[0*nk+0] = 1 # B00
            # C10 += 1
            w[0*ni+1] = 1 # C10
            U.append(u); V.append(v); W.append(w)
            
            # Term 3: M3 = A00(B01-B11)
            u = [0]*ni*nj
            v = [0]*nj*nk
            w = [0]*nk*ni
            u[0*ni+0] = 1 # A00
            v[0*nk+1] = 1 # B01
            v[1*nk+1] = -1 # B11
            # C01 += 1
            w[1*ni+0] = 1 # C01
            U.append(u); V.append(v); W.append(w)
            
            # Term 4: M4 = A11(B10-B00)
            u = [0]*ni*nj
            v = [0]*nj*nk
            w = [0]*nk*ni
            u[1*ni+1] = 1 # A11
            v[1*nk+0] = 1 # B10
            v[0*nk+0] = -1 # B00
            # C10 += 1, C00 += 1
            w[0*ni+1] = 1 # C10
            w[0*ni+0] = 1 # C00
            U.append(u); V.append(v); W.append(w)
            
            # Term 5: M5 = (A00+A01)B11
            u = [0]*ni*nj
            v = [0]*nj*nk
            w = [0]*nk*ni
            u[0*ni+0] = 1 # A00
            u[0*ni+1] = 1 # A01
            v[1*nk+1] = 1 # B11
            # C00 -= 1, C01 += 1
            w[0*ni+0] = -1 # C00
            w[1*ni+0] = 1 # C01
            U.append(u); V.append(v); W.append(w)
            
            # Term 6: M6 = (A10-A00)(B00+B01)
            u = [0]*ni*nj
            v = [0]*nj*nk
            w = [0]*nk*ni
            u[1*ni+0] = 1 # A10
            u[0*ni+0] = -1 # A00
            v[0*nk+0] = 1 # B00
            v[0*nk+1] = 1 # B01
            # C11 += 1
            w[1*ni+1] = 1 # C11
            U.append(u); V.append(v); W.append(w)
            
            # Term 7: M7 = (A01-A11)(B10+B11)
            u = [0]*ni*nj
            v = [0]*nj*nk
            w = [0]*nk*ni
            u[0*ni+1] = 1 # A01
            u[1*ni+1] = -1 # A11
            v[1*nk+0] = 1 # B10
            v[1*nk+1] = 1 # B11
            # C00 += 1, C11 -= 1
            w[0*ni+0] = 1 # C00
            w[1*ni+1] = -1 # C11
            U.append(u); V.append(v); W.append(w)
            
        else:
            # Schoolbook
            for li in range(ni):
                for lj in range(nj):
                    for lk in range(nk):
                        u = [0]*ni*nj
                        v = [0]*nj*nk
                        w = [0]*nk*ni
                        u[li*nj+lj] = 1
                        v[lj*nk+lk] = 1
                        w[lk*ni+li] = 1
                        U.append(u); V.append(v); W.append(w)
                        
    # Iterate over blocks
    for bi, (ms, mo) in enumerate(zip(m_splits, m_offsets)):
        for bk, (ps, po) in enumerate(zip(p_splits, p_offsets)):
            use_strassen = (ms == 2 and ps == 2)
            add_block(0, n, mo, mo+ms, po, po+ps, use_strassen)
            
    # Convert to numpy arrays or lists of lists
    # The prompt expects lists of lists or similar
    rank = len(U)
    U_mat = [[0]*rank for _ in range(n*m)]
    V_mat = [[0]*rank for _ in range(m*p)]
    W_mat = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            if U[r][i] != 0:
                U_mat[i][r] = U[r][i]
        for j in range(m*p):
            if V[r][j] != 0:
                V_mat[j][r] = V[r][j]
        for k in range(p*n):
            if W[r][k] != 0:
                W_mat[k][r] = W[r][k]
                
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
