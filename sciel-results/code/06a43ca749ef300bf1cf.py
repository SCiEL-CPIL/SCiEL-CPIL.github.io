"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Partitions
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2, 1]
    
    # Calculate offsets
    n_offsets = [0]
    for x in n_parts: n_offsets.append(n_offsets[-1] + x)
    m_offsets = [0]
    for x in m_parts: m_offsets.append(m_offsets[-1] + x)
    p_offsets = [0]
    for x in p_parts: p_offsets.append(p_offsets[-1] + x)
    
    # Pre-compute Strassen 2x2x2 decomposition
    # U: 4x7, V: 4x7, W: 4x7
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 1], # i=0, j=0
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1
        [0, 1, 0, 0, 0, 1, 0],  # i=1, j=0
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0
        [0, 0, 0, 1, 0, 1, 0],  # j=0, k=1
        [0, 0, 1, 0, 0, 0, 1],  # j=1, k=0
        [1, 0, -1, 0, 1, 0, 1]  # j=1, k=1
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], # k=0, i=0
        [0, 1, 0, 1, 0, 0, 0],  # k=0, i=1
        [0, 0, 1, 0, 1, 0, 0],  # k=1, i=0
        [1, -1, 1, 0, 0, 1, 0]  # k=1, i=1
    ]
    
    # Function to get decomposition for n,m,p
    def get_decomp(n, m, p):
        if n == 2 and m == 2 and p == 2:
            return U_strassen, V_strassen, W_strassen, 7
        else:
            rank = n * m * p
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            return U, V, W, rank

    # Main construction
    # Total rank is sum of ranks of blocks
    total_rank = 0
    block_ranks = []
    for ni in n_parts:
        for mj in m_parts:
            for pk in p_parts:
                _, _, _, r = get_decomp(ni, mj, pk)
                block_ranks.append(r)
                total_rank += r
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    rank_offset = 0
    for idx_n, ni in enumerate(n_parts):
        i_start = n_offsets[idx_n]
        for idx_m, mj in enumerate(m_parts):
            j_start = m_offsets[idx_m]
            for idx_p, pk in enumerate(p_parts):
                k_start = p_offsets[idx_p]
                
                Ub, Vb, Wb, rb = get_decomp(ni, mj, pk)
                
                # Place Ub into U
                for i in range(ni):
                    for j in range(mj):
                        u_row = i_start + i
                        u_col_base = j_start + j
                        # U index is (i*m + j)
                        # But wait, U is indexed by (i,j) of the FULL matrix.
                        # So row is i_start + i, but the column index in U is (i*m + j)?
                        # No, U is a matrix of shape (n*m, rank).
                        # The row index in U corresponds to the pair (i,j).
                        # So row = i_start + i? No.
                        # The global index for A is i, j.
                        # i is global i. j is global j.
                        # So row in U is i * m + j.
                        # i is i_start + i_local.
                        # j is j_start + j_local.
                        row_u = (i_start + i) * m + (j_start + j)
                        
                        for r in range(rb):
                            U[row_u][rank_offset + r] = Ub[i*mj + j][r] # Ub index is i_local * mj + j_local?
                            # Wait, Ub is for the sub-matrix of size ni x mj.
                            # So Ub row index is i_local * mj + j_local.
                            # Let's fix the indexing.
                            
                # Re-do placement carefully
                # Reset U, V, W? No, we are adding.
                # We need to map local indices to global.
                
                for i_loc in range(ni):
                    for j_loc in range(mj):
                        # Global i, j
                        i_glob = i_start + i_loc
                        j_glob = j_start + j_loc
                        # U row index
                        u_row = i_glob * m + j_glob
                        # Local U row index
                        u_loc_row = i_loc * mj + j_loc
                        
                        for r in range(rb):
                            U[u_row][rank_offset + r] = Ub[u_loc_row][r]
                            
                for j_loc in range(mj):
                    for k_loc in range(pk):
                        j_glob = j_start + j_loc
                        k_glob = k_start + k_loc
                        v_row = j_glob * p + k_glob
                        v_loc_row = j_loc * pk + k_loc
                        
                        for r in range(rb):
                            V[v_row][rank_offset + r] = Vb[v_loc_row][r]
                            
                for k_loc in range(pk):
                    for i_loc in range(ni):
                        k_glob = k_start + k_loc
                        i_glob = i_start + i_loc
                        w_row = k_glob * n + i_glob
                        w_loc_row = k_loc * ni + i_loc
                        
                        for r in range(rb):
                            W[w_row][rank_offset + r] = Wb[w_loc_row][r]
                            
                rank_offset += rb

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
