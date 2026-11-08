"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    Uses a Kronecker-product style decomposition:
    T[2,3,9] is the direct sum of three T[2,3,3] tensors.
    Currently uses the schoolbook decomposition for each T[2,3,3] block.
    Rank = 3 * (2*3*3) = 54.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 3
    p_block = 3
    
    # Define the base decomposition for a 2x3 * 3x3 block.
    # This is the schoolbook decomposition.
    # U_block: 6 x 18
    # V_block: 9 x 18
    # W_block: 6 x 18 (Wait, W is p*n = 3*2=6)
    
    # We will generate the columns for the base block first.
    base_U_cols = []
    base_V_cols = []
    base_W_cols = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p_block):
                u_idx = i * m + j
                v_idx = j * p_block + k
                w_idx = k * n + i
                
                col_u = [0] * (n * m)
                col_v = [0] * (m * p_block)
                col_w = [0] * (p_block * n)
                
                col_u[u_idx] = 1
                col_v[v_idx] = 1
                col_w[w_idx] = 1
                
                base_U_cols.append(col_u)
                base_V_cols.append(col_v)
                base_W_cols.append(col_w)
    
    base_rank = len(base_U_cols)
    
    # Assemble the full matrices by repeating the base block for each of the 3 blocks.
    # U is the same for all blocks (it only depends on the 2x3 input).
    # Actually, U has shape (n*m, total_rank).
    # The U part is identical for each block, but the columns are distinct in the global rank.
    # So we just repeat the base_U columns 3 times?
    # No, the tensor is a direct sum.
    # T = T1 + T2 + T3.
    # The U matrix for the sum is [U1, U2, U3] horizontally.
    # Since U1=U2=U3 (same input dimensions), we repeat the columns.
    
    U = []
    V = []
    W = []
    
    # Build U
    # U has n*m rows.
    # Each row is a concatenation of the row from each block's U.
    for r in range(n * m):
        row_u = []
        for b in range(num_blocks):
            for c in range(base_rank):
                row_u.append(base_U_cols[c][r])
        U.append(row_u)
        
    # Build V
    # V has m*p rows.
    # The rows are grouped by the block index in the second dimension.
    # For block b, the rows corresponding to that block use base_V columns.
    # Other blocks have 0 in those rows.
    for r in range(m * p):
        row_v = []
        # Determine which block this row belongs to
        # r = j*p + k_global
        # k_global = b*p_block + k_local
        # So we need to map r to b and k_local.
        # Actually, it's easier to iterate blocks and fill.
        # But we are building row by row.
        # Let's just use the logic:
        # If r is in the range for block b, then it uses base_V columns for that block.
        # Which block?
        # The rows of V are indexed by (j, k_global).
        # k_global determines the block.
        k_global = r % p
        b = k_global // p_block
        k_local = k_global % p_block
        j = r // p
        
        # The base_V column index for this (j, k_local) is:
        # In base_V, index is j*p_block + k_local.
        # But base_V_cols are indexed by the triple (i,j,k).
        # We need the column corresponding to the specific (j, k_local) in the base decomposition.
        # In the base decomposition, each (i,j,k) is a separate column.
        # Wait, the schoolbook decomposition has a column for EVERY (i,j,k).
        # So for a fixed (j, k_local), there are n columns (one for each i).
        # In the global V, for a fixed row (j, k_global), the entries in the columns corresponding to block b should be 1 if the column's k matches k_local and j matches j.
        # This is getting complicated to assemble row by row from the base columns.
        
        # Let's stick to the column-by-column assembly which was correct in the original code.
        pass
        
    # Revert to column assembly for correctness and simplicity.
    cols_U = []
    cols_V = []
    cols_W = []
    
    for b in range(num_blocks):
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    u_idx = i * m + j
                    v_idx = j * p + (b * p_block + k)
                    w_idx = (b * p_block + k) * n + i
                    
                    col_u = [0] * (n * m)
                    col_v = [0] * (m * p)
                    col_w = [0] * (p * n)
                    
                    col_u[u_idx] = 1
                    col_v[v_idx] = 1
                    col_w[w_idx] = 1
                    
                    cols_U.append(col_u)
                    cols_V.append(col_v)
                    cols_W.append(col_w)
                    
    rank = len(cols_U)
    
    U = [[cols_U[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[cols_V[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[cols_W[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
