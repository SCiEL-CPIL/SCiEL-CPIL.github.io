"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build a block decomposition.
    # Split n=3 into 2+1
    # Split m=4 into 2+2
    # Split p=5 into 2+2+1
    
    # Blocks:
    # (n=2, m=2, p=2): 4 blocks. Use Strassen (rank 7).
    # (n=2, m=2, p=1): 2 blocks. Use Rank 1 (all ones).
    # (n=1, m=2, p=2): 4 blocks. Use Rank 1 (all ones).
    # (n=1, m=2, p=1): 2 blocks. Use Rank 1 (all ones).
    
    # Total rank = 4*7 + 2*1 + 4*1 + 2*1 = 28 + 2 + 4 + 2 = 36.
    
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term (all ones in the block)
    def add_rank1_block(n_loc, m_loc, p_loc, u_off, v_off, w_off):
        nonlocal col
        for i in range(n_loc):
            for j in range(m_loc):
                for k in range(p_loc):
                    u_idx = u_off + i * m_loc + j
                    v_idx = v_off + j * p_loc + k
                    w_idx = w_off + k * n_loc + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
        col += 1

    # Helper to add Strassen decomposition for 2x2x2
    def add_strassen_2x2x2(u_off, v_off, w_off):
        nonlocal col
        # Indices in 2x2 matrix: 00, 01, 10, 11
        # U local indices: 0, 1, 2, 3
        # V local indices: 0, 1, 2, 3
        # W local indices: 0, 1, 2, 3 (corresponding to C00, C01, C10, C11)
        
        # Term 1: (a+d)(e+h) -> W: 1, 0, 0, 1
        U[u_off + 0][col] = 1; U[u_off + 3][col] = 1
        V[v_off + 0][col] = 1; V[v_off + 3][col] = 1
        W[w_off + 0][col] = 1; W[w_off + 3][col] = 1
        col += 1
        
        # Term 2: (c+d)e -> W: 0, 0, 1, -1
        U[u_off + 2][col] = 1; U[u_off + 3][col] = 1
        V[v_off + 0][col] = 1
        W[w_off + 2][col] = 1; W[w_off + 3][col] = -1
        col += 1
        
        # Term 3: a(f-h) -> W: 0, 1, 0, 1
        U[u_off + 0][col] = 1
        V[v_off + 1][col] = 1; V[v_off + 3][col] = -1
        W[w_off + 1][col] = 1; W[w_off + 3][col] = 1
        col += 1
        
        # Term 4: d(g-e) -> W: 1, 0, 1, 0
        U[u_off + 3][col] = 1
        V[v_off + 2][col] = 1; V[v_off + 0][col] = -1
        W[w_off + 0][col] = 1; W[w_off + 2][col] = 1
        col += 1
        
        # Term 5: (a+b)h -> W: -1, 1, 0, 0
        U[u_off + 0][col] = 1; U[u_off + 1][col] = 1
        V[v_off + 3][col] = 1
        W[w_off + 0][col] = -1; W[w_off + 1][col] = 1
        col += 1
        
        # Term 6: (c-a)(e+f) -> W: 0, 0, 0, 1
        U[u_off + 2][col] = 1; U[u_off + 0][col] = -1
        V[v_off + 0][col] = 1; V[v_off + 1][col] = 1
        W[w_off + 3][col] = 1
        col += 1
        
        # Term 7: (b-d)(g+h) -> W: 1, 0, 0, 0
        U[u_off + 1][col] = 1; U[u_off + 3][col] = -1
        V[v_off + 2][col] = 1; V[v_off + 3][col] = 1
        W[w_off + 0][col] = 1
        col += 1

    # Iterate over blocks
    # n splits: [2, 1]
    # m splits: [2, 2]
    # p splits: [2, 2, 1]
    
    n_splits = [2, 1]
    m_splits = [2, 2]
    p_splits = [2, 2, 1]
    
    u_offset = 0
    for n_i, n_loc in enumerate(n_splits):
        v_offset = 0
        for m_i, m_loc in enumerate(m_splits):
            w_offset = 0
            for p_i, p_loc in enumerate(p_splits):
                if n_loc == 2 and m_loc == 2 and p_loc == 2:
                    add_strassen_2x2x2(u_offset, v_offset, w_offset)
                else:
                    add_rank1_block(n_loc, m_loc, p_loc, u_offset, v_offset, w_offset)
                w_offset += p_loc * n_loc
            v_offset += m_loc * p
        u_offset += n_loc * m
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
