"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_strassen_2x2x2(a_base, b_base, c_base, c_size):
        # a_base, b_base, c_base are indices into the global U, V, W arrays
        # c_size is the size of the C block (2x2)
        # We need to map the local 2x2 indices to global indices
        
        # Local indices for A (2x2): (0,0), (0,1), (1,0), (1,1)
        # Local indices for B (2x2): (0,0), (0,1), (1,0), (1,1)
        # Local indices for C (2x2): (0,0), (0,1), (1,0), (1,1)
        
        # Strassen multipliers:
        # m1 = (a+d)(e+h) -> C11 + C22
        # m2 = (c+d)e     -> C21
        # m3 = a(f-h)     -> C12
        # m4 = d(g-e)     -> C11 + C21
        # m5 = (a+b)h     -> C11 + C12
        # m6 = (c-a)(e+f) -> C22
        # m7 = (b-d)(g+h) -> C11 + C22
        
        # We need to construct the vectors for each term.
        
        # Helper to create a vector of length n*m with 1s at specific (i,j) positions
        def make_u(inds):
            vec = [0] * (n*m)
            for i, j in inds:
                vec[i*m + j] = 1
            return vec
            
        def make_v(inds):
            vec = [0] * (m*p)
            for j, k in inds:
                vec[j*p + k] = 1
            return vec
            
        def make_w(inds):
            vec = [0] * (p*n)
            for k, i in inds:
                vec[k*n + i] = 1
            return vec

        # Define local to global mapping
        # a_base is (i_start, j_start)
        # b_base is (j_start, k_start)
        # c_base is (i_start, k_start)
        
        i_s, j_s = a_base
        j_bs, k_s = b_base
        i_cs, k_cs = c_base
        
        # Check consistency
        assert j_s == j_bs, "A and B inner dims must match"
        assert i_s == i_cs, "A and C row dims must match"
        assert k_s == k_cs, "B and C col dims must match"

        # Local indices
        # A: (0,0)->(i_s, j_s), (0,1)->(i_s, j_s+1), (1,0)->(i_s+1, j_s), (1,1)->(i_s+1, j_s+1)
        # B: (0,0)->(j_s, k_s), (0,1)->(j_s, k_s+1), (1,0)->(j_s+1, k_s), (1,1)->(j_s+1, k_s+1)
        # C: (0,0)->(i_s, k_s), (0,1)->(i_s, k_s+1), (1,0)->(i_s+1, k_s), (1,1)->(i_s+1, k_s+1)
        
        # m1: (A00+A11) * (B00+B11) -> C00 + C11
        u1 = make_u([(i_s, j_s), (i_s+1, j_s+1)])
        v1 = make_v([(j_s, k_s), (j_s+1, k_s+1)])
        w1 = make_w([(k_s, i_s), (k_s+1, i_s+1)])
        add_term(u1, v1, w1)
        
        # m2: (A10+A11) * B00 -> C10
        u2 = make_u([(i_s+1, j_s), (i_s+1, j_s+1)])
        v2 = make_v([(j_s, k_s)])
        w2 = make_w([(k_s, i_s+1)])
        add_term(u2, v2, w2)
        
        # m3: A00 * (B01-B11) -> C01
        u3 = make_u([(i_s, j_s)])
        v3 = make_v([(j_s, k_s+1), (j_s+1, k_s+1)])
        v3[j_s*p + k_s+1] = 1
        v3[(j_s+1)*p + k_s+1] = -1
        w3 = make_w([(k_s+1, i_s)])
        add_term(u3, v3, w3)
        
        # m4: A11 * (B10-B00) -> C00 + C10
        u4 = make_u([(i_s+1, j_s+1)])
        v4 = make_v([(j_s+1, k_s), (j_s, k_s)])
        v4[(j_s+1)*p + k_s] = 1
        v4[j_s*p + k_s] = -1
        w4 = make_w([(k_s, i_s), (k_s, i_s+1)])
        add_term(u4, v4, w4)
        
        # m5: (A00+A01) * B11 -> C00 + C01
        u5 = make_u([(i_s, j_s), (i_s, j_s+1)])
        v5 = make_v([(j_s+1, k_s+1)])
        w5 = make_w([(k_s, i_s), (k_s+1, i_s)])
        add_term(u5, v5, w5)
        
        # m6: (A10-A00) * (B00+B01) -> C11
        u6 = make_u([(i_s+1, j_s), (i_s, j_s)])
        u6[(i_s+1)*m + j_s] = 1
        u6[i_s*m + j_s] = -1
        v6 = make_v([(j_s, k_s), (j_s, k_s+1)])
        w6 = make_w([(k_s+1, i_s+1)])
        add_term(u6, v6, w6)
        
        # m7: (A01-A11) * (B10+B11) -> C00 + C11
        u7 = make_u([(i_s, j_s+1), (i_s+1, j_s+1)])
        u7[i_s*m + j_s+1] = 1
        u7[(i_s+1)*m + j_s+1] = -1
        v7 = make_v([(j_s+1, k_s), (j_s+1, k_s+1)])
        w7 = make_w([(k_s, i_s), (k_s+1, i_s+1)])
        add_term(u7, v7, w7)

    def add_schoolbook_block(a_base, b_base, c_base, a_h, a_w, b_h, b_w):
        # a_h x a_w block in A
        # b_h x b_w block in B
        # c_h x c_w block in C (c_h=a_h, c_w=b_w, a_w=b_h)
        
        i_s, j_s = a_base
        j_bs, k_s = b_base
        i_cs, k_cs = c_base
        
        for ai in range(a_h):
            for aj in range(a_w):
                for bk in range(b_w):
                    # Term: A[i_s+ai, j_s+aj] * B[j_s+aj, k_s+bk] -> C[i_s+ai, k_s+bk]
                    u = [0] * (n*m)
                    u[(i_s+ai)*m + (j_s+aj)] = 1
                    
                    v = [0] * (m*p)
                    v[(j_s+aj)*p + (k_s+bk)] = 1
                    
                    w = [0] * (p*n)
                    w[(k_s+bk)*n + (i_s+ai)] = 1
                    
                    add_term(u, v, w)

    # Partitioning
    # n=3: rows 0-1 (2x), row 2 (1x)
    # m=5: cols 0-1 (2x), cols 2-3 (2x), col 4 (1x)
    # p=6: cols 0-1 (2x), cols 2-3 (2x), cols 4-5 (2x)
    
    # Block grid for C (3x6)
    # C blocks:
    # (0,0) 2x2, (0,1) 2x2, (0,2) 2x2
    # (1,0) 1x2, (1,1) 1x2, (1,2) 1x2
    
    # A blocks:
    # (0,0) 2x2, (0,1) 2x2, (0,2) 2x1
    # (1,0) 1x2, (1,1) 1x2, (1,2) 1x1
    
    # B blocks:
    # (0,0) 2x2, (0,1) 2x2, (0,2) 2x2
    # (1,0) 2x2, (1,1) 2x2, (1,2) 2x2
    # (2,0) 1x2, (2,1) 1x2, (2,2) 1x2
    
    # Iterate C block positions
    # c_row_block, c_col_block
    # c_row_block 0: i=0..1, c_row_block 1: i=2
    # c_col_block 0: k=0..1, c_col_block 1: k=2..3, c_col_block 2: k=4..5
    
    for cb_r in range(2): # 0, 1
        for cb_c in range(3): # 0, 1, 2
            
            # Determine C block base
            if cb_r == 0:
                i_start_c = 0
                c_h = 2
            else:
                i_start_c = 2
                c_h = 1
                
            if cb_c == 0:
                k_start_c = 0
            elif cb_c == 1:
                k_start_c = 2
            else:
                k_start_c = 4
            c_w = 2
            
            # Sum contributions from all A_col_block, B_row_block pairs
            # A has 3 col blocks (0,1,2) corresponding to m partitions (2,2,1)
            # B has 3 row blocks (0,1,2) corresponding to m partitions (2,2,1)
            
            for ab_c in range(3): # A column block
                for bb_r in range(3): # B row block
                    if ab_c != bb_r:
                        continue # Inner dimension must match
                    
                    # Get A block details
                    if ab_c == 0:
                        j_start_a = 0
                        a_w_block = 2
                    elif ab_c == 1:
                        j_start_a = 2
                        a_w_block = 2
                    else:
                        j_start_a = 4
                        a_w_block = 1
                        
                    # Get B block details
                    if bb_r == 0:
                        j_start_b = 0
                        b_h_block = 2
                    elif bb_r == 1:
                        j_start_b = 2
                        b_h_block = 2
                    else:
                        j_start_b = 4
                        b_h_block = 1
                        
                    # Get A row block details (depends on cb_r)
                    if cb_r == 0:
                        i_start_a = 0
                        a_h_block = 2
                    else:
                        i_start_a = 2
                        a_h_block = 1
                        
                    # Get B col block details (depends on cb_c)
                    if cb_c == 0:
                        k_start_b = 0
                    elif cb_c == 1:
                        k_start_b = 2
                    else:
                        k_start_b = 4
                    b_w_block = 2
                    
                    # Check if we can use Strassen
                    # Requires a_h_block=2, a_w_block=2, b_h_block=2, b_w_block=2
                    if a_h_block == 2 and a_w_block == 2 and b_h_block == 2 and b_w_block == 2:
                        add_strassen_2x2x2((i_start_a, j_start_a), (j_start_b, k_start_b), (i_start_c, k_start_c), 2)
                    else:
                        add_schoolbook_block((i_start_a, j_start_a), (j_start_b, k_start_b), (i_start_c, k_start_c), a_h_block, a_w_block, b_h_block, b_w_block)
                        
    rank = len(U_cols)
    U = [U_cols[i] for i in range(rank)]
    V = [V_cols[i] for i in range(rank)]
    W = [W_cols[i] for i in range(rank)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
