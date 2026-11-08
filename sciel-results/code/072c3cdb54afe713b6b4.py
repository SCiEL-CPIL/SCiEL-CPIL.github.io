"""Program for matrix multiplication 3x3 by 3x7 using Laderman's algorithm."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # We decompose the 7 columns into:
    # - Two blocks of 3 columns each (using Laderman's algorithm, rank 23 each)
    # - One block of 1 column (using schoolbook, rank 9)
    # Total rank: 23 + 23 + 9 = 55
    
    rank_total = 55
    
    # Initialize U, V, W as lists of lists
    # U: (n*m) x rank_total = 9 x 55
    # V: (m*p) x rank_total = 21 x 55
    # W: (p*n) x rank_total = 21 x 55
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Laderman's algorithm for 3x3x3 matrix multiplication
    # We define the 23 intermediate products and their coefficients
    # For a 3x3x3 block, we have:
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Laderman's 23 multiplications (from the algorithm):
    # The coefficients are derived from the Laderman algorithm.
    # Let me define them explicitly.
    
    # For a 3x3x3 block starting at column offset col_offset in the B matrix,
    # and row offset row_offset in the C matrix:
    
    def add_laderman_block(col_offset, row_offset, rank_offset):
        """Add Laderman's 23-multiplication block to U, V, W."""
        # Laderman's algorithm coefficients
        # For 3x3x3, the 23 products are:
        
        # Define the 23 (U, V, W) coefficient triples
        # Each entry is (u_coeffs, v_coeffs, w_coeffs) where each is a list of (index, coeff)
        
        # Laderman's algorithm (standard form):
        # Let A = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
        # Let B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
        # C = A * B
        
        # The 23 intermediate products:
        # M0 = (a00 + a11 + a22) * (b00 + b11 + b22)
        # M1 = (a01 + a12 + a20) * (b01 + b12 + b20)
        # M2 = (a02 + a10 + a21) * (b02 + b10 + b21)
        # M3 = (a00 + a01 + a02) * (b00 + b10 + b20)
        # M4 = (a10 + a11 + a12) * (b01 + b11 + b21)
        # M5 = (a20 + a21 + a22) * (b02 + b12 + b22)
        # M6 = (a00 + a10 + a20) * (b00 + b01 + b02)
        # M7 = (a01 + a11 + a21) * (b10 + b11 + b12)
        # M8 = (a02 + a12 + a22) * (b20 + b21 + b22)
        # M9 = (a00 - a11 + a22) * (b00 - b11 + b22)
        # M10 = (a01 - a12 + a20) * (b01 - b12 + b20)
        # M11 = (a02 - a10 + a21) * (b02 - b10 + b21)
        # M12 = (a00 + a11 - a22) * (b00 + b11 - b22)
        # M13 = (a01 + a12 - a20) * (b01 + b12 - b20)
        # M14 = (a02 + a10 - a21) * (b02 + b10 - b21)
        # M15 = (a00 - a01 + a02) * (b00 - b10 + b20)
        # M16 = (a10 - a11 + a12) * (b01 - b11 + b21)
        # M17 = (a20 - a21 + a22) * (b02 - b12 + b22)
        # M18 = (a00 + a01 - a02) * (b00 + b10 - b20)
        # M19 = (a10 + a11 - a12) * (b01 + b11 - b21)
        # M20 = (a20 + a21 - a22) * (b02 + b12 - b22)
        # M21 = (a00 - a10 + a20) * (b00 - b01 + b02)
        # M22 = (a01 - a11 + a21) * (b10 - b11 + b12)
        
        # Actually, let me use the standard Laderman decomposition.
        # The standard Laderman algorithm uses 23 multiplications.
        # Let me look up the exact coefficients.
        
        # From the literature, Laderman's algorithm for 3x3x3:
        # The 23 bilinear forms are:
        
        # I'll use a known correct set of coefficients.
        # Let me define them as (u_indices, v_indices, w_indices, sign)
        
        # Actually, let me just use the schoolbook for all blocks to ensure correctness,
        # but that gives rank 63. To improve, I need a valid low-rank decomposition.
        
        # Let me try a different approach: use Strassen-like decomposition for 2x2 blocks
        # and combine with the remaining parts.
        
        # Actually, let me just implement the schoolbook decomposition correctly
        # and accept rank 63, since I cannot guarantee Laderman's coefficients.
        pass
    
    # Since I cannot guarantee the exact Laderman coefficients from memory,
    # let me use a valid decomposition. The schoolbook gives rank 63.
    # But I can do better by using a known low-rank decomposition.
    
    # Let me try to construct a rank 55 decomposition using a combination
    # of known algorithms.
    
    # Actually, let me just use the schoolbook decomposition to ensure validity.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
