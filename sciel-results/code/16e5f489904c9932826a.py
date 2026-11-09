"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p is 9, decomposed into 3 blocks of size 3.
    num_blocks = p // 3
    p_block = 3
    
    cols_U = []
    cols_V = []
    cols_W = []
    
    # For each block of the 3x9 matrix (which is 3 blocks of 3x3)
    for b in range(num_blocks):
        # For each element in the 2x3 * 3x3 multiplication
        # A is 2x3 (indices i, j)
        # B is 3x3 (indices j, k)
        # C is 2x3 (indices i, k)
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    # Global indices
                    # U (A): i*m + j
                    u_idx = i * m + j
                    
                    # V (B): j*p + (b*p_block + k)
                    v_idx = j * p + (b * p_block + k)
                    
                    # W (C): (b*p_block + k)*n + i
                    # Note: C is 2x9. Flattened row-major?
                    # Task says W[k*n+i, r].
                    # k_global is the column of C (0..8).
                    # i is the row of C (0..1).
                    # So index is k_global * n + i.
                    k_global = b * p_block + k
                    w_idx = k_global * n + i
                    
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
    
    # Convert list of columns to list of rows (matrix)
    U = [[cols_U[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[cols_V[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[cols_W[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
