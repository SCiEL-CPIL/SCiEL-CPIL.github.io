"""Tensor decomposition for 2x2 by 2x14 matrix multiplication using Strassen per block."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p = 14, so we have 7 blocks of size 2x2
    num_blocks = p // 2  # 7
    rank = 7 * num_blocks  # 49
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 49
    V = [[0] * rank for _ in range(m * p)]   # 28 x 49
    W = [[0] * rank for _ in range(p * n)]   # 28 x 49
    
    # Strassen decomposition for 2x2 * 2x2 -> 2x2
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B
    #
    # M1 = A00 * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = (A00 + A01) * B11
    # M4 = A11 * (B10 - B00)
    # M5 = A00 * (B01 - B11)
    # M6 = (A10 + A00) * B11
    # M7 = (A01 + A11) * (B00 - B10)
    #
    # C00 = M1 + M4 - M3 + M5
    # C10 = M2 + M4
    # C01 = M3 - M5
    # C11 = M1 - M2 + M6 + M7
    
    for blk in range(num_blocks):
        # Block uses B columns 2*blk and 2*blk+1
        # B[0, 2*blk] = B00, B[1, 2*blk] = B10
        # B[0, 2*blk+1] = B01, B[1, 2*blk+1] = B11
        
        b00_idx = 0 * p + 2 * blk      # B[0, 2*blk]
        b10_idx = 1 * p + 2 * blk      # B[1, 2*blk]
        b01_idx = 0 * p + (2 * blk + 1) # B[0, 2*blk+1]
        b11_idx = 1 * p + (2 * blk + 1) # B[1, 2*blk+1]
        
        # C indices: W[k*n + i, r] where k is column of C, i is row of C
        # C[0, 2*blk] -> W[2*blk * n + 0]
        # C[1, 2*blk] -> W[2*blk * n + 1]
        # C[0, 2*blk+1] -> W[(2*blk+1) * n + 0]
        # C[1, 2*blk+1] -> W[(2*blk+1) * n + 1]
        
        c00_idx = (2 * blk) * n + 0
        c10_idx = (2 * blk) * n + 1
        c01_idx = (2 * blk + 1) * n + 0
        c11_idx = (2 * blk + 1) * n + 1
        
        base = blk * 7
        
        # M1 = A00 * (B00 + B11)
        r = base + 0
        U[0][r] = 1  # A00
        V[b00_idx][r] = 1
        V[b11_idx][r] = 1
        W[c00_idx][r] = 1
        W[c11_idx][r] = 1
        
        # M2 = (A10 + A11) * B00
        r = base + 1
        U[2][r] = 1  # A10
        U[3][r] = 1  # A11
        V[b00_idx][r] = 1
        W[c10_idx][r] = 1
        W[c11_idx][r] = -1
        
        # M3 = (A00 + A01) * B11
        r = base + 2
        U[0][r] = 1  # A00
        U[1][r] = 1  # A01
        V[b11_idx][r] = 1
        W[c00_idx][r] = -1
        W[c01_idx][r] = 1
        
        # M4 = A11 * (B10 - B00)
        r = base + 3
        U[3][r] = 1  # A11
        V[b10_idx][r] = 1
        V[b00_idx][r] = -1
        W[c00_idx][r] = 1
        W[c10_idx][r] = 1
        
        # M5 = A00 * (B01 - B11)
        r = base + 4
        U[0][r] = 1  # A00
        V[b01_idx][r] = 1
        V[b11_idx][r] = -1
        W[c00_idx][r] = 1
        W[c01_idx][r] = -1
        
        # M6 = (A10 + A00) * B11
        r = base + 5
        U[2][r] = 1  # A10
        U[0][r] = 1  # A00
        V[b11_idx][r] = 1
        W[c11_idx][r] = 1
        
        # M7 = (A01 + A11) * (B00 - B10)
        r = base + 6
        U[1][r] = 1  # A01
        U[3][r] = 1  # A11
        V[b00_idx][r] = 1
        V[b10_idx][r] = -1
        W[c11_idx][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
