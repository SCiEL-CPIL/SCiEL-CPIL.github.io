import numpy as np

DIMENSIONS = [4, 5, 6]

def _strassen_2x2():
    """Return the 7 Strassen terms for 2x2x2.
    Each term is (u_indices, v_indices, w_indices) where indices are (row, col) for A, (col, row) for B, (row, col) for C.
    A is 2x2, B is 2x2, C = A*B is 2x2.
    U: shape (4, 7) - indexed by (a_row*2+a_col, term)
    V: shape (4, 7) - indexed by (b_col*2+b_row, term)  [B stored as B[col][row]? No, B is 2x2, B[col,row]]
    W: shape (4, 7) - indexed by (c_row*2+c_col, term)
    """
    # Strassen terms for C = A*B:
    # M1 = (A00+A11)(B01+B10)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A01)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each term r, we need:
    # U[(a_row,a_col), r] and V[(b_col,b_row), r] and W[(c_row,c_col), r]
    # such that sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = C[i,k] for the 2x2 case
    
    # Let's define the 7 terms. For 2x2:
    # n=2, m=2, p=2
    # U index: i*2+j where i in {0,1}, j in {0,1}
    # V index: j*2+k where j in {0,1}, k in {0,1}
    # W index: k*2+i where k in {0,1}, i in {0,1}
    
    terms = []
    
    # M1 = (A00+A11)(B01+B10)
    # U: A00 and A11 get +1
    # V: B01 and B10 get +1
    # W: C00 and C11 get +1
    terms.append({(0,0):1, (1,1):1}, {(0,1):1, (1,0):1}, {(0,0):1, (1,1):1})
    
    # M2 = (A10+A11)B00
    terms.append({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1})
    
    # M3 = A00(B10-B11)
    terms.append({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1})
    
    # M4 = A11(B01-B00)
    terms.append({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1})
    
    # M5 = (A00+A10)B11
    terms.append({(0,0):1, (1,0):1}, {(1,1):1}, {(0,0):-1, (0,1):1})
    
    # M6 = (A11-A01)(B00+B10)
    terms.append({(1,1):1, (0,1):-1}, {(0,0):1, (1,0):1}, {(1,1):1})
    
    # M7 = (A01-A11)(B01+B11)
    terms.append({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(0,0):1, (1,1):-1})
    
    return terms

def _build_strassen_4x4x4():
    """Build Strassen decomposition for 4x4x4.
    Uses 2x2 Strassen applied to 2x2 blocks.
    Returns (U, V, W) each of shape (16, 49) where:
    U indexed by (i*4+j, r) for i,j in 0..3
    V indexed by (j*4+k, r) for j,k in 0..3
    W indexed by (k*4+i, r) for k,i in 0..3
    """
    s2x2 = _strassen_2x2()
    # 7 terms for 2x2, each with 7 sub-terms -> 49 total terms
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # The 4x4 matrix A is split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # Similarly B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    
    # For each of the 7 Strassen terms at the block level,
    # we apply 2x2 Strassen within each 2x2 sub-problem.
    
    # Block-level Strassen:
    # M1 = (A00+A11)(B01+B10) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2
    # M3 = A00(B10-B11) -> C01 += M3
    # M4 = A11(B01-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A10)B11 -> C00 -= M5, C01 += M5
    # M6 = (A11-A01)(B00+B10) -> C11 += M6
    # M7 = (A01-A11)(B01+B11) -> C00 += M7, C11 -= M7
    
    # For each block-level term, the "A" part is a 2x2 matrix (sum of 2x2 blocks),
    # and the "B" part is a 2x2 matrix (sum of 2x2 blocks).
    # We apply 2x2 Strassen to each.
    
    # Let's enumerate: for each block term bt (0..6), and each sub-term st (0..6),
    # we have a combined term index = bt*7 + st.
    
    # For block term bt, we need to determine:
    # - Which 2x2 A-blocks contribute and with what signs
    # - Which 2x2 B-blocks contribute and with what signs
    # - Which 2x2 C-blocks receive the result and with what signs
    
    # Block term definitions: (a_blocks, b_blocks, c_blocks)
    # a_blocks: list of (block_idx, sign) where block_idx: 0=A00,1=A01,2=A10,3=A11
    # b_blocks: list of (block_idx, sign) where block_idx: 0=B00,1=B01,2=B10,3=B11
    # c_blocks: list of (block_idx, sign) where block_idx: 0=C00,1=C01,2=C10,3=C11
    
    block_terms = [
        # M1
        ([(0,1),(3,1)], [(1,1),(2,1)], [(0,1),(3,1)]),
        # M2
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),
        # M3
        ([(0,1)], [(2,1),(3,-1)], [(1,1)]),
        # M4
        ([(3,1)], [(1,1),(0,-1)], [(0,1),(2,1)]),
        # M5
        ([(0,1),(2,1)], [(3,1)], [(0,-1),(1,1)]),
        # M6
        ([(3,1),(1,-1)], [(0,1),(2,1)], [(3,1)]),
        # M7
        ([(1,1),(3,-1)], [(1,1),(3,1)], [(0,1),(3,-1)]),
    ]
    
    for bt in range(7):
        a_blocks, b_blocks, c_blocks = block_terms[bt]
        
        for st in range(7):
            r = bt * 7 + st
            s2x2_term = s2x2[st]
            u_2x2, v_2x2, w_2x2 = s2x2_term
            
            # For each A-block contribution, for each entry in u_2x2:
            for (ab_idx, ab_sign) in a_blocks:
                # ab_idx: 0=A00, 1=A01, 2=A10, 3=A11
                # A00: rows 0-1, cols 0-1
                # A01: rows 0-1, cols 2-3
                # A10: rows 2-3, cols 0-1
                # A11: rows 2-3, cols 2-3
                ar = (ab_idx // 2) * 2  # row offset of block
                ac = (ab_idx % 2) * 2   # col offset of block
                
                for ((ui, uj), uv) in u_2x2.items():
                    # ui, uj are local indices in the 2x2 block
                    global_i = ar + ui
                    global_j = ac + uj
                    U[global_i * 4 + global_j, r] = ab_sign * uv
            
            # For each B-block contribution, for each entry in v_2x2:
            for (bb_idx, bb_sign) in b_blocks:
                # B00: rows 0-1, cols 0-1
                # B01: rows 0-1, cols 2-3
                # B10: rows 2-3, cols 0-1
                # B11: rows 2-3, cols 2-3
                br = (bb_idx // 2) * 2  # row offset (which is col of B in the product)
                bc = (bb_idx % 2) * 2   # col offset (which is row of B in the product)
                
                for ((vj, vk), vv) in v_2x2.items():
                    # vj, vk are local indices: vj is col of B, vk is row of B
                    # In our indexing, V is indexed by (j*4+k) where j is col, k is row
                    global_j = br + vj
                    global_k = bc + vk
                    V[global_j * 4 + global_k, r] = bb_sign * vv
            
            # For each C-block contribution, for each entry in w_2x2:
            for (cb_idx, cb_sign) in c_blocks:
                # C00: rows 0-1, cols 0-1
                # C01: rows 0-1, cols 2-3
                # C10: rows 2-3, cols 0-1
                # C11: rows 2-3, cols 2-3
                cr = (cb_idx // 2) * 2  # row offset
                cc = (cb_idx % 2) * 2   # col offset
                
                for ((wk, wi), wv) in w_2x2.items():
                    # wk, wi are local indices: wk is col of C (which is k), wi is row of C (which is i)
                    # W is indexed by (k*4+i)
                    global_k = cr + wk
                    global_i = cc + wi
                    W[global_k * 4 + global_i, r] = cb_sign * wv
    
    return U, V, W

def _build_schoolbook_4x4x2():
    """Schoolbook for 4x4x2. A is 4x4, B is 4x2, C is 4x2.
    Returns (U, V, W) each of shape (16, 32) for U, (8, 32) for V, (8, 32) for W.
    U: (i*4+j, r), V: (j*4+k, r), W: (k*4+i, r)
    """
    n, m, p = 4, 4, 2
    rank = n * m * p  # 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _build_schoolbook_4x1x6():
    """Schoolbook for 4x1x6. A is 4x1, B is 1x6, C is 4x6.
    Returns (U, V, W) each of shape (4, 24) for U, (6, 24) for V, (24, 24) for W.
    U: (i*1+j, r) = (i, r), V: (j*6+k, r) = (k, r), W: (k*4+i, r)
    """
    n, m, p = 4, 1, 6
    rank = n * m * p  # 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Split: A is 4x5, B is 5x6
    # Split A into A1 (4x4) and A2 (4x1)
    # Split B into B1 (4x6) and B2 (1x6)
    # C = A1*B1 + A2*B2
    # Split B1 into B1a (4x4) and B1b (4x2)
    # C = A1*B1a + A1*B1b + A2*B2
    # Term 1: A1*B1a is 4x4x4 -> Strassen rank 49
    # Term 2: A1*B1b is 4x4x2 -> Schoolbook rank 32
    # Term 3: A2*B2 is 4x1x6 -> Schoolbook rank 24
    # Total rank: 49 + 32 + 24 = 105
    
    # Wait, let me reconsider. The total rank should be lower.
    # Actually, I think I can do better by being more careful.
    
    # Let me just use the straightforward block decomposition.
    # Total rank = 49 + 32 + 24 = 105.
    # That's better than 120.
    
    # Build each part
    U1, V1, W1 = _build_strassen_4x4x4()  # shapes (16,49), (16,49), (16,49)
    U2, V2, W2 = _build_schoolbook_4x4x2()  # shapes (16,32), (8,32), (8,32)
    U3, V3, W3 = _build_schoolbook_4x1x6()  # shapes (4,24), (6,24), (24,24)
    
    total_rank = 49 + 32 + 24  # 105
    
    # Now we need to map these into the full 4x5x6 problem.
    # Full problem: A is 4x5, B is 5x6, C is 4x6
    # U_full: shape (4*5, total_rank) = (20, 105)
    # V_full: shape (5*6, total_rank) = (30, 105)
    # W_full: shape (6*4, total_rank) = (24, 105)
    
    U_full = np.zeros((20, total_rank))
    V_full = np.zeros((30, total_rank))
    W_full = np.zeros((24, total_rank))
    
    # Term 1: A1*B1a (4x4x4)
    # A1 = A[:, 0:4], B1a = B[0:4, 0:4]
    # U1 is indexed by (i*4+j) for i in 0..3, j in 0..3
    # In full U, this is (i*5+j) for i in 0..3, j in 0..3
    for i in range(4):
        for j in range(4):
            U_full[i*5 + j, 0:49] = U1[i*4 + j, :]
    
    # V1 is indexed by (j*4+k) for j in 0..3, k in 0..3
    # In full V, this is (j*6+k) for j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            V_full[j*6 + k, 0:49] = V1[j*4 + k, :]
    
    # W1 is indexed by (k*4+i) for k in 0..3, i in 0..3
    # In full W, this is (k*4+i) for k in 0..3, i in 0..3
    for k in range(4):
        for i in range(4):
            W_full[k*4 + i, 0:49] = W1[k*4 + i, :]
    
    # Term 2: A1*B1b (4x4x2)
    # A1 = A[:, 0:4], B1b = B[0:4, 4:6]
    # U2 is indexed by (i*4+j) for i in 0..3, j in 0..3
    for i in range(4):
        for j in range(4):
            U_full[i*5 + j, 49:81] = U2[i*4 + j, :]
    
    # V2 is indexed by (j*4+k) for j in 0..3, k in 0..1
    # In full V, B1b has cols 4,5, so k in 0..1 maps to k+4
    for j in range(4):
        for k in range(2):
            V_full[j*6 + (k+4), 49:81] = V2[j*4 + k, :]
    
    # W2 is indexed by (k*4+i) for k in 0..1, i in 0..3
    # In full W, k in 0..1 maps to k+4
    for k in range(2):
        for i in range(4):
            W_full[(k+4)*4 + i, 49:81] = W2[k*4 + i, :]
    
    # Term 3: A2*B2 (4x1x6)
    # A2 = A[:, 4], B2 = B[4, :]
    # U3 is indexed by (i*1+0) = i for i in 0..3
    for i in range(4):
        U_full[i*5 + 4, 81:105] = U3[i, :]
    
    # V3 is indexed by (0*6+k) = k for k in 0..5
    for k in range(6):
        V_full[4*6 + k, 81:105] = V3[k, :]
    
    # W3 is indexed by (k*4+i) for k in 0..5, i in 0..3
    for k in range(6):
        for i in range(4):
            W_full[k*4 + i, 81:105] = W3[k*4 + i, :]
    
    # Convert to lists for the return interface
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
