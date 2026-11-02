"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-21 decomposition using Strassen's algorithm for each of the three 2x2x2 blocks.
    
    The 2x2 * 2x6 multiplication decomposes into three independent 2x2 * 2x2 multiplications.
    Each 2x2x2 uses Strassen's 7-rank decomposition.
    Total rank = 3 * 7 = 21.
    
    U has shape (n*m, rank) = (4, 21)
    V has shape (m*p, rank) = (12, 21)  
    W has shape (p*n, rank) = (12, 21)
    
    Indexing:
    - U[i*m + j, c] where i in [0,2), j in [0,2)
    - V[j*p + k, c] where j in [0,2), k in [0,6)
    - W[k*n + i, c] where k in [0,6), i in [0,2)
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 21
    V = [[0] * rank for _ in range(m * p)]   # 12 x 21
    W = [[0] * rank for _ in range(p * n)]   # 12 x 21
    
    # Strassen's 7 intermediate products for 2x2 * 2x2:
    # M1 = (A00)(B00 + B11)
    # M2 = (A10 + A11)(B00)
    # M3 = (A00)(B10)
    # M4 = (A11)(B01 + B11)
    # M5 = (A00 + A11)(B11)
    # M6 = (A10 + A00)(B11 + B10)
    # M7 = (A11 + A10)(B01 + B00)
    #
    # Output:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M1 + M2 - M5 - M7
    # C11 = M2 + M3 - M1 + M6
    
    # For each block b in [0, 1, 2], the B matrix columns are [2b, 2b+1]
    # A matrix rows: A[0,:] = row 0 (i=0), A[1,:] = row 1 (i=1)
    # B matrix columns: B[:,0] = col 2b (k=2b), B[:,1] = col 2b+1 (k=2b+1)
    # C matrix: C[0,2b] = C00, C[0,2b+1] = C01, C[1,2b] = C10, C[1,2b+1] = C11
    
    for b in range(3):
        base_col = b * 7  # starting column index for this block's 7 Strassen terms
        
        # B column indices for this block
        k0 = 2 * b      # first column of B in this block
        k1 = 2 * b + 1  # second column of B in this block
        
        # M1 = (A00)(B00 + B11)
        # U: A00 -> U[0*2+0, base_col+0] = U[0, base_col+0] = 1
        # V: B00 -> V[0*6+k0, base_col+0] = V[k0, base_col+0] = 1
        #     B11 -> V[1*6+k1, base_col+0] = V[6+k1, base_col+0] = 1
        # W: C00 gets +1, C10 gets +1
        # C00 is at W[k0*n + 0, ...] = W[2b, ...]
        # C10 is at W[k0*n + 1, ...] = W[2b+1, ...]
        U[0][base_col + 0] = 1
        V[k0][base_col + 0] = 1
        V[6 + k1][base_col + 0] = 1
        W[2 * b][base_col + 0] = 1       # C00 += M1
        W[2 * b + 1][base_col + 0] = 1   # C10 += M1
        
        # M2 = (A10 + A11)(B00)
        # U: A10 -> U[1*2+0, base_col+1] = U[2, base_col+1] = 1
        #     A11 -> U[1*2+1, base_col+1] = U[3, base_col+1] = 1
        # V: B00 -> V[0*6+k0, base_col+1] = V[k0, base_col+1] = 1
        # W: C10 gets +1, C11 gets +1
        # C10 is at W[2b+1, ...], C11 is at W[2b+1+1, ...] = W[2b+2, ...]
        # Wait, C11 is at output position (1, 2b+1), so W[(2b+1)*n + 1, ...] = W[2b+1+1, ...] = W[2b+2, ...]
        U[2][base_col + 1] = 1
        U[3][base_col + 1] = 1
        V[k0][base_col + 1] = 1
        W[2 * b + 1][base_col + 1] = 1   # C10 += M2
        W[2 * b + 2][base_col + 1] = 1   # C11 += M2
        
        # M3 = (A00)(B10)
        # U: A00 -> U[0, base_col+2] = 1
        # V: B10 -> V[1*6+k0, base_col+2] = V[6+k0, base_col+2] = 1
        # W: C01 gets +1, C11 gets +1
        # C01 is at W[(2b+1)*n + 0, ...] = W[2b+1, ...]
        # C11 is at W[(2b+1)*n + 1, ...] = W[2b+2, ...]
        U[0][base_col + 2] = 1
        V[6 + k0][base_col + 2] = 1
        W[2 * b + 1][base_col + 2] = 1   # C01 += M3
        W[2 * b + 2][base_col + 2] = 1   # C11 += M3
        
        # M4 = (A11)(B01 + B11)
        # U: A11 -> U[3, base_col+3] = 1
        # V: B01 -> V[0*6+k1, base_col+3] = V[k1, base_col+3] = 1
        #     B11 -> V[1*6+k1, base_col+3] = V[6+k1, base_col+3] = 1
        # W: C00 gets -1, C01 gets +1
        # C00 is at W[2b, ...], C01 is at W[2b+1, ...]
        U[3][base_col + 3] = 1
        V[k1][base_col + 3] = 1
        V[6 + k1][base_col + 3] = 1
        W[2 * b][base_col + 3] = -1      # C00 -= M4
        W[2 * b + 1][base_col + 3] = 1   # C01 += M4
        
        # M5 = (A00 + A11)(B11)
        # U: A00 -> U[0, base_col+4] = 1
        #     A11 -> U[3, base_col+4] = 1
        # V: B11 -> V[1*6+k1, base_col+4] = V[6+k1, base_col+4] = 1
        # W: C00 gets +1, C10 gets -1
        # C00 is at W[2b, ...], C10 is at W[2b+1, ...]
        U[0][base_col + 4] = 1
        U[3][base_col + 4] = 1
        V[6 + k1][base_col + 4] = 1
        W[2 * b][base_col + 4] = 1       # C00 += M5
        W[2 * b + 1][base_col + 4] = -1  # C10 -= M5
        
        # M6 = (A10 + A00)(B11 + B10)
        # U: A10 -> U[2, base_col+5] = 1
        #     A00 -> U[0, base_col+5] = 1
        # V: B11 -> V[1*6+k1, base_col+5] = V[6+k1, base_col+5] = 1
        #     B10 -> V[1*6+k0, base_col+5] = V[6+k0, base_col+5] = 1
        # W: C11 gets +1
        # C11 is at W[2b+2, ...]
        U[2][base_col + 5] = 1
        U[0][base_col + 5] = 1
        V[6 + k1][base_col + 5] = 1
        V[6 + k0][base_col + 5] = 1
        W[2 * b + 2][base_col + 5] = 1   # C11 += M6
        
        # M7 = (A11 + A10)(B01 + B00)
        # U: A11 -> U[3, base_col+6] = 1
        #     A10 -> U[2, base_col+6] = 1
        # V: B01 -> V[0*6+k1, base_col+6] = V[k1, base_col+6] = 1
        #     B00 -> V[0*6+k0, base_col+6] = V[k0, base_col+6] = 1
        # W: C00 gets +1, C10 gets -1
        # C00 is at W[2b, ...], C10 is at W[2b+1, ...]
        U[3][base_col + 6] = 1
        U[2][base_col + 6] = 1
        V[k1][base_col + 6] = 1
        V[k0][base_col + 6] = 1
        W[2 * b][base_col + 6] = 1       # C00 += M7
        W[2 * b + 1][base_col + 6] = -1  # C10 -= M7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
