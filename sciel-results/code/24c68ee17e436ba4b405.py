"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    # Rank 97: 49 (Strassen for 4x4x4) + 48 (Schoolbook for 4x4x3)
    rank = 97
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 4x4x4 decomposition
    # 7 Strassen products, each of rank 7
    # Total 49 terms
    
    # Define the 7 Strassen terms for 2x2x2
    # u, v, w are vectors of length 4 (flattened 2x2)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_terms = []
    # M1 = (X00+X11)(Y00+Y11) -> Z00=1, Z11=1
    strassen_terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (X10+X11)Y00 -> Z10=1
    strassen_terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
    # M3 = X00(Y01-Y11) -> Z01=1
    strassen_terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
    # M4 = X11(Y10-Y00) -> Z10=1
    strassen_terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 0]))
    # M5 = (X00+X01)Y11 -> Z01=1
    strassen_terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]))
    # M6 = (X10-X00)(Y00+Y01) -> Z11=1
    strassen_terms.append([[-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]])
    # M7 = (X01-X11)(Y10+Y11) -> Z00=1
    strassen_terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    
    # Strassen products for 4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    # C blocks: C00 (0,0), C01 (0,1), C10 (1,0), C11 (1,1)
    
    # Mapping of Strassen product to A blocks, B blocks, C blocks
    # Each block is 2x2.
    # We need to add the 2x2 terms to the global 4x4 matrices.
    
    def add_strassen_term(col, a_blocks, b_blocks, c_blocks, u, v, w):
        # a_blocks: list of (block_row, block_col, sign)
        # b_blocks: list of (block_row, block_col, sign)
        # c_blocks: list of (block_row, block_col, sign)
        
        for br, bc, sa in a_blocks:
            for i in range(2):
                for j in range(2):
                    ui = u[i*2+j]
                    if ui != 0:
                        U[br*2+i][bc*2+j][col] += sa * ui
        for br, bc, sb in b_blocks:
            for j in range(2):
                for k in range(2):
                    vk = v[j*2+k]
                    if vk != 0:
                        V[br*2+j][bc*2+k][col] += sb * vk
        for br, bc, sc in c_blocks:
            for k in range(2):
                for i in range(2):
                    wi = w[k*2+i]
                    if wi != 0:
                        W[bc*2+k][br*2+i][col] += sc * wi # W index is k*n+i? 
                        # W shape is (p*n, rank). p=4, n=4.
                        # Index is k*4 + i.
                        # In c_blocks, br is row of C (i), bc is col of C (k).
                        # So i=br, k=bc.
                        # W index is k*n + i = bc*4 + br.
                        pass

    # Correct W index logic:
    # W[k*n + i, r]
    # In add_strassen_term, c_blocks gives (row_C, col_C) = (i, k).
    # So W index is k*4 + i.
    
    def add_strassen_term_correct(col, a_blocks, b_blocks, c_blocks, u, v, w):
        for br, bc, sa in a_blocks:
            for i in range(2):
                for j in range(2):
                    ui = u[i*2+j]
                    if ui != 0:
                        U[br*2+i][bc*2+j][col] += sa * ui
        for br, bc, sb in b_blocks:
            for j in range(2):
                for k in range(2):
                    vk = v[j*2+k]
                    if vk != 0:
                        V[br*2+j][bc*2+k][col] += sb * vk
        for br, bc, sc in c_blocks:
            for i in range(2):
                for k in range(2):
                    # u, v, w are 2x2 vectors.
                    # w is indexed by (row_C, col_C) = (i, k).
                    # w_flat[i*2+k]
                    wi = w[i*2+k]
                    if wi != 0:
                        # W index is k*n + i. n=4.
                        # Global i = br*2 + i_local
                        # Global k = bc*2 + k_local
                        # Wait, c_blocks (br, bc) are block indices.
                        # i_local, k_local are local indices.
                        # Global i = br*2 + i_local
                        # Global k = bc*2 + k_local
                        # W index = k_global * n + i_global
                        # But W is for the whole 4x4x4 part?
                        # Yes, for the first 49 terms.
                        # The C matrix is 4x4.
                        # So i_global, k_global are in 0..3.
                        W[bc*2+k][br*2+i][col] += sc * wi
                        # Wait, W index is k*n + i.
                        # k is col_C, i is row_C.
                        # So W[k*4 + i].
                        # In the loop, i is row_local, k is col_local.
                        # Global row i_g = br*2 + i
                        # Global col k_g = bc*2 + k
                        # W index = k_g * 4 + i_g
                        # My code above: W[bc*2+k][br*2+i] is WRONG.
                        # It should be W[k_g * 4 + i_g]
                        pass

    # Let's rewrite add_strassen_term_correct properly.
    
    def add_term(col, a_blocks, b_blocks, c_blocks, u, v, w):
        # a_blocks: list of (row_block, col_block, sign)
        for rb, cb, s in a_blocks:
            for i in range(2):
                for j in range(2):
                    val = u[i*2+j]
                    if val != 0:
                        U[rb*2+i][cb*2+j][col] += s * val
        for rb, cb, s in b_blocks:
            for j in range(2):
                for k in range(2):
                    val = v[j*2+k]
                    if val != 0:
                        V[rb*2+j][cb*2+k][col] += s * val
        for rb, cb, s in c_blocks:
            for i in range(2):
                for k in range(2):
                    val = w[i*2+k]
                    if val != 0:
                        # W index: k*n + i
                        # n=4
                        # Global i = rb*2 + i
                        # Global k = cb*2 + k
                        idx = (cb*2+k)*4 + (rb*2+i)
                        W[idx][col] += s * val

    # M1 = (A00+A11)(B00+B11) -> C00, C11
    a_blocks = [(0,0,1), (1,1,1)]
    b_blocks = [(0,0,1), (1,1,1)]
    c_blocks = [(0,0,1), (1,1,1)]
    
    # M2 = (A10+A11)B00 -> C10
    a_blocks = [(1,0,1), (1,1,1)]
    b_blocks = [(0,0,1)]
    c_blocks = [(1,0,1)]
    
    # M3 = A00(B01-B11) -> C01
    a_blocks = [(0,0,1)]
    b_blocks = [(0,1,1), (1,1,-1)]
    c_blocks = [(0,1,1)]
    
    # M4 = A11(B10-B00) -> C10
    a_blocks = [(1,1,1)]
    b_blocks = [(1,0,1), (0,0,-1)]
    c_blocks = [(1,0,1)]
    
    # M5 = (A00+A01)B11 -> C01
    a_blocks = [(0,0,1), (0,1,1)]
    b_blocks = [(1,1,1)]
    c_blocks = [(0,1,1)]
    
    # M6 = (A10-A00)(B00+B01) -> C11
    a_blocks = [(1,0,1), (0,0,-1)]
    b_blocks = [(0,0,1), (0,1,1)]
    c_blocks = [(1,1,1)]
    
    # M7 = (A01-A11)(B10+B11) -> C00
    a_blocks = [(0,1,1), (1,1,-1)]
    b_blocks = [(1,0,1), (1,1,1)]
    c_blocks = [(0,0,1)]
    
    all_m_blocks = [
        (a_blocks, b_blocks, c_blocks),
        (a_blocks, b_blocks, c_blocks), # Placeholder, need to define for each M
    ]
    
    # Re-define properly
    M_defs = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]), # M1
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]), # M2
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]), # M3
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(1,0,1)]), # M4
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1)]), # M5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]), # M6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]), # M7
    ]
    
    col = 0
    for m_idx in range(7):
        a_b, b_b, c_b = M_defs[m_idx]
        for t_idx in range(7):
            u, v, w = strassen_terms[t_idx]
            add_term(col, a_b, b_b, c_b, u, v, w)
            col += 1
            
    # Schoolbook 4x4x3
    # A is 4x4, B is 4x3
    # Rank 48
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i*4+j][col] = 1
                V[j*4+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
