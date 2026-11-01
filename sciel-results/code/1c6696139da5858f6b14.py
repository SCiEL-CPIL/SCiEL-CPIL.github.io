"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    """Rank-14 decomposition using Strassen's algorithm applied to two 2x2 blocks."""
    n, m, p = DIMENSIONS  # 2, 2, 4
    
    # Strassen's 7 products for 2x2 * 2x2
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # U indices: i*m+j where i is row of A (0,1), j is col of A (0,1)
    # V indices: j*p+k where j is row of B (0,1), k is col of B (0,1,2,3)
    # W indices: k*n+i where k is col of result (0,1,2,3), i is row of result (0,1)
    
    # For block 1 (cols 0,1 of B): B indices are j*4+k where k in {0,1}
    # For block 2 (cols 2,3 of B): B indices are j*4+k where k in {2,3}
    
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14
    
    # Strassen products for a 2x2 * 2x2 block
    # Each product has coefficients in U (for A), V (for B), W (for C)
    # A indices: 00->0, 01->1, 10->2, 11->3 (i*m+j)
    # B indices for block offset b: 00->b, 01->b+1, 10->4+b, 11->4+b+1 (j*4+k)
    # C indices for block offset c: 00->c, 01->c+1, 10->2+c, 11->3+c (k*2+i)
    
    def add_strassen_block(b_offset, c_offset, col_start):
        """Add 7 Strassen products for B block at column offset b_offset, C at c_offset."""
        # M1 = (A00+A11)(B00+B11)
        U[0][col_start + 0] += 1  # A00
        U[3][col_start + 0] += 1  # A11
        V[b_offset][col_start + 0] += 1       # B00
        V[4 + b_offset + 1][col_start + 0] += 1  # B11
        W[c_offset][col_start + 0] += 1       # C00
        W[2 + c_offset][col_start + 0] += 1   # C10
        
        # M2 = (A10+A11)B00
        U[2][col_start + 1] += 1  # A10
        U[3][col_start + 1] += 1  # A11
        V[b_offset][col_start + 1] += 1       # B00
        W[2 + c_offset][col_start + 1] += 1   # C10
        
        # M3 = A00(B01-B11)
        U[0][col_start + 2] += 1  # A00
        V[b_offset + 1][col_start + 2] += 1   # B01
        V[4 + b_offset + 1][col_start + 2] -= 1  # -B11
        W[c_offset + 1][col_start + 2] += 1   # C01
        
        # M4 = A11(B10-B00)
        U[3][col_start + 3] += 1  # A11
        V[4 + b_offset][col_start + 3] += 1   # B10
        V[b_offset][col_start + 3] -= 1       # -B00
        W[c_offset][col_start + 3] += 1       # C00
        W[2 + c_offset][col_start + 3] += 1   # C10
        
        # M5 = (A00+A10)B11
        U[0][col_start + 4] += 1  # A00
        U[2][col_start + 4] += 1  # A10
        V[4 + b_offset + 1][col_start + 4] += 1  # B11
        W[c_offset][col_start + 4] -= 1   # -C00
        W[c_offset + 1][col_start + 4] += 1   # C01
        
        # M6 = (A11-A00)(B00+B10)
        U[3][col_start + 5] += 1  # A11
        U[0][col_start + 5] -= 1  # -A00
        V[b_offset][col_start + 5] += 1       # B00
        V[4 + b_offset][col_start + 5] += 1   # B10
        W[3 + c_offset][col_start + 5] += 1   # C11
        
        # M7 = (A01-A11)(B01+B11)
        U[1][col_start + 6] += 1  # A01
        U[3][col_start + 6] -= 1  # -A11
        V[b_offset + 1][col_start + 6] += 1   # B01
        V[4 + b_offset + 1][col_start + 6] += 1  # B11
        W[c_offset][col_start + 6] += 1       # C00
        W[3 + c_offset][col_start + 6] += 1   # C11
    
    # Block 1: B columns 0,1 -> b_offset=0, C columns 0,1 -> c_offset=0
    add_strassen_block(0, 0, 0)
    # Block 2: B columns 2,3 -> b_offset=2, C columns 2,3 -> c_offset=2
    add_strassen_block(2, 2, 7)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
