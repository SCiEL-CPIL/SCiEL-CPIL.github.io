"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Block decomposition
    # n=4 -> 2 blocks of 2
    # p=6 -> 3 blocks of 2
    # m=5 -> 3 blocks: 2, 2, 1
    
    row_blocks = [0, 2] # start indices, size 2
    col_blocks = [0, 2, 4] # start indices, sizes 2, 2, 2 (for p)
    mid_blocks = [0, 2, 4] # start indices, sizes 2, 2, 1 (for m)
    
    # Pre-allocate U, V, W with 0s
    # We don't know total rank yet, so we will build lists of columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a local block
    def add_block(ib, kb, mb, U_local, V_local, W_local):
        # ib: row block index (0 or 1)
        # kb: col block index (0, 1, or 2)
        # mb: mid block index (0, 1, or 2)
        
        n_local = 2
        m_local = mid_blocks[mb+1] - mid_blocks[mb] if mb < 2 else 1 # Actually mid_blocks sizes are 2,2,1
        # Let's define sizes explicitly
        mid_sizes = [2, 2, 1]
        m_local = mid_sizes[mb]
        p_local = 2
        
        # Local U is (n_local * m_local, rank_local)
        # Local V is (m_local * p_local, rank_local)
        # Local W is (p_local * n_local, rank_local)
        
        rank_local = U_local.shape[1]
        
        for r in range(rank_local):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # Map U
            for i_l in range(n_local):
                for j_l in range(m_local):
                    i_g = row_blocks[ib] + i_l
                    j_g = mid_blocks[mb] + j_l
                    u_col[i_g * m + j_g] = U_local[i_l * m_local + j_l, r]
            
            # Map V
            for j_l in range(m_local):
                for k_l in range(p_local):
                    j_g = mid_blocks[mb] + j_l
                    k_g = col_blocks[kb] + k_l
                    v_col[j_g * p + k_g] = V_local[j_l * p_local + k_l, r]
            
            # Map W
            for k_l in range(p_local):
                for i_l in range(n_local):
                    k_g = col_blocks[kb] + k_l
                    i_g = row_blocks[ib] + i_l
                    w_col[k_g * n + i_g] = W_local[k_l * n_local + i_l, r]
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    # Generate Strassen block for 2x2x2
    def get_strassen_block():
        # Indices: i in 0..1, j in 0..1, k in 0..1
        # U: 4x7, V: 4x7, W: 4x7
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22) -> C11, C22
        # U: A11(0,0)=1, A22(1,1)=1 -> idx 0, 3
        U[0,0] = 1; U[3,0] = 1
        # V: B11(0,0)=1, B22(1,1)=1 -> idx 0, 3
        V[0,0] = 1; V[3,0] = 1
        # W: C11(0,0)=1, C22(1,1)=1 -> idx 0, 3
        W[0,0] = 1; W[3,0] = 1
        
        # M2 = (A21+A22)B11 -> C21
        # U: A21(1,0)=1, A22(1,1)=1 -> idx 2, 3
        U[2,1] = 1; U[3,1] = 1
        # V: B11(0,0)=1 -> idx 0
        V[0,1] = 1
        # W: C21(1,0)=1 -> idx 2
        W[2,1] = 1
        
        # M3 = A11(B12-B22) -> C12
        # U: A11(0,0)=1 -> idx 0
        U[0,2] = 1
        # V: B12(0,1)=1, B22(1,1)=-1 -> idx 1, 3
        V[1,2] = 1; V[3,2] = -1
        # W: C12(0,1)=1 -> idx 1
        W[1,2] = 1
        
        # M4 = A22(B21-B11) -> C11, C21
        # U: A22(1,1)=1 -> idx 3
        U[3,3] = 1
        # V: B21(1,0)=1, B11(0,0)=-1 -> idx 2, 0
        V[2,3] = 1; V[0,3] = -1
        # W: C11(0,0)=1, C21(1,0)=1 -> idx 0, 2
        W[0,3] = 1; W[2,3] = 1
        
        # M5 = (A11+A12)B22 -> C11, C12
        # U: A11(0,0)=1, A12(0,1)=1 -> idx 0, 1
        U[0,4] = 1; U[1,4] = 1
        # V: B22(1,1)=1 -> idx 3
        V[3,4] = 1
        # W: C11(0,0)=1, C12(0,1)=1 -> idx 0, 1
        W[0,4] = 1; W[1,4] = 1
        
        # M6 = (A21-A11)(B11+B12) -> C22
        # U: A21(1,0)=1, A11(0,0)=-1 -> idx 2, 0
        U[2,5] = 1; U[0,5] = -1
        # V: B11(0,0)=1, B12(0,1)=1 -> idx 0, 1
        V[0,5] = 1; V[1,5] = 1
        # W: C22(1,1)=1 -> idx 3
        W[3,5] = 1
        
        # M7 = (A12-A22)(B21+B22) -> C11, C22
        # U: A12(0,1)=1, A22(1,1)=-1 -> idx 1, 3
        U[1,6] = 1; U[3,6] = -1
        # V: B21(1,0)=1, B22(1,1)=1 -> idx 2, 3
        V[2,6] = 1; V[3,6] = 1
        # W: C11(0,0)=1, C22(1,1)=1 -> idx 0, 3
        W[0,6] = 1; W[3,6] = 1
        
        return U, V, W

    # Generate Schoolbook block for 2x1x2
    def get_schoolbook_block_2x1x2():
        # i in 0..1, j in 0..0, k in 0..1
        # U: 2x4, V: 2x4, W: 4x4
        U = [[0]*4 for _ in range(2)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(4)]
        
        # Term 0: A11 B11 -> C11
        U[0,0] = 1
        V[0,0] = 1
        W[0,0] = 1 # C11 (k=0, i=0) -> idx 0*2+0=0
        
        # Term 1: A11 B12 -> C12
        U[0,1] = 1
        V[1,1] = 1 # B12 (j=0, k=1) -> idx 0*2+1=1
        W[1,1] = 1 # C12 (k=1, i=0) -> idx 1*2+0=2? 
        # Wait, W index is k*n + i. n_local=2.
        # C12: k=1, i=0. idx = 1*2 + 0 = 2.
        W[2,1] = 1
        
        # Term 2: A21 B11 -> C21
        U[1,2] = 1 # A21 (i=1, j=0) -> idx 1
        V[0,2] = 1
        W[2,2] = 1 # C21 (k=0, i=1) -> idx 0*2+1=1? 
        # C21: k=0, i=1. idx = 0*2 + 1 = 1.
        W[1,2] = 1
        
        # Term 3: A21 B12 -> C22
        U[1,3] = 1
        V[1,3] = 1
        W[3,3] = 1 # C22 (k=1, i=1) -> idx 1*2+1=3.
        W[3,3] = 1
        
        return U, V, W

    # Iterate over blocks
    for ib in range(2):
        for kb in range(3):
            # Determine middle block size
            # We have 3 middle blocks: 2, 2, 1
            # But kb is col block (0,1,2).
            # The middle dimension is shared.
            # We should iterate over middle blocks too?
            # No, the decomposition is:
            # C[ib, kb] = A[ib, :] * B[:, kb]
            # A[ib, :] is 2x5. B[:, kb] is 5x2.
            # We split the 5 into 2, 2, 1.
            # So for each (ib, kb), we have 3 sub-sub-blocks?
            # No, that would mean we sum over the middle parts.
            # The tensor for the whole multiplication is the sum of the tensors for each middle part.
            # So we iterate over mb as well.
            pass

    # Correct iteration:
    U_cols = []
    V_cols = []
    W_cols = []
    
    for ib in range(2):
        for kb in range(3):
            for mb in range(3):
                mid_size = [2, 2, 1][mb]
                if mid_size == 2:
                    U_loc, V_loc, W_loc = get_strassen_block()
                else:
                    U_loc, V_loc, W_loc = get_schoolbook_block_2x1x2()
                
                # Add block
                # U_loc shape: (2*mid_size, rank)
                # V_loc shape: (mid_size*2, rank)
                # W_loc shape: (2*2, rank)
                
                rank_loc = len(U_loc[0])
                
                for r in range(rank_loc):
                    u_col = [0] * (n * m)
                    v_col = [0] * (m * p)
                    w_col = [0] * (p * n)
                    
                    # Map U
                    for i_l in range(2):
                        for j_l in range(mid_size):
                            i_g = ib * 2 + i_l
                            j_g = [0, 2, 4][mb] + j_l
                            u_col[i_g * m + j_g] = U_loc[i_l * mid_size + j_l][r]
                    
                    # Map V
                    for j_l in range(mid_size):
                        for k_l in range(2):
                            j_g = [0, 2, 4][mb] + j_l
                            k_g = kb * 2 + k_l
                            v_col[j_g * p + k_g] = V_loc[j_l * 2 + k_l][r]
                    
                    # Map W
                    for k_l in range(2):
                        for i_l in range(2):
                            k_g = kb * 2 + k_l
                            i_g = ib * 2 + i_l
                            w_col[k_g * n + i_g] = W_loc[k_l * 2 + i_l][r]
                    
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)

    rank = len(U_cols)
    # Transpose to (n*m, rank)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
