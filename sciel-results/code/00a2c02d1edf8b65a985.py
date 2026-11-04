"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-98 decomposition for 4x4x8 matrix multiplication.
    Split the 8 columns into two 4-column blocks.
    Each block is a 4x4x4 multiplication, decomposed with recursive Strassen (rank 49).
    Total rank = 49 + 49 = 98.
    """
    n, m, p = DIMENSIONS
    
    def strassen_2x2_terms():
        """Return the 7 rank-1 terms for 2x2x2 Strassen decomposition."""
        # Each term: (u, v, w) where u, v, w are 4-element lists [row*2+col]
        terms = []
        # M1 = (a00+a11)(b00+b11)
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2 = (a10+a11)b00
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1]))
        # M3 = a00(b01-b11)
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [1, 0, 0, 0]))
        # M4 = a11(b10-b00)
        terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]))
        # M5 = (a00+a01)b11
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]))
        # M6 = (a10-a00)(b00+b01)
        terms.append(([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 0]))
        # M7 = (a01-a11)(b10+b11)
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, -1, 0]))
        return terms
    
    base_2x2 = strassen_2x2_terms()
    
    # The 7 Strassen products for 4x4x4 (viewed as 2x2 blocks of 2x2 matrices)
    # Each product: (u_blocks, v_blocks, w_blocks)
    # where u_blocks[i*2+j] is the coefficient for the (i,j) 2x2 block of A
    strassen_products = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1 = (A00+A11)(B00+B11)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1]),    # M2 = (A10+A11)B00
        ([1, 0, 0, 0], [0, 1, 0, -1], [1, 0, 0, 0]),   # M3 = A00(B01-B11)
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]),   # M4 = A11(B10-B00)
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),    # M5 = (A00+A01)B11
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 0]),   # M6 = (A10-A00)(B00+B01)
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, -1, 0]),  # M7 = (A01-A11)(B10+B11)
    ]
    
    # Construct the 49 rank-1 terms for 4x4x4
    rank_4x4_terms = []
    for (u_b, v_b, w_b) in strassen_products:
        for (u_2, v_2, w_2) in base_2x2:
            u_4x4 = [0] * 16
            v_4x4 = [0] * 16
            w_4x4 = [0] * 16
            
            for block in range(4):
                ub = u_b[block]
                vb = v_b[block]
                wb = w_b[block]
                
                for local in range(4):
                    if ub != 0:
                        u_4x4[block * 4 + local] += ub * u_2[local]
                    if vb != 0:
                        v_4x4[block * 4 + local] += vb * v_2[local]
                    if wb != 0:
                        w_4x4[block * 4 + local] += wb * w_2[local]
            
            rank_4x4_terms.append((u_4x4, v_4x4, w_4x4))
    
    # Now construct the 4x4x8 decomposition with rank 98
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # Block 1: columns 0-3 of B and C
    for t in range(49):
        col = t
        u_4x4, v_4x4, w_4x4 = rank_4x4_terms[t]
        
        for idx in range(16):
            U[idx][col] = u_4x4[idx]
            V[idx][col] = v_4x4[idx]
            W[idx][col] = w_4x4[idx]
    
    # Block 2: columns 4-7 of B and C
    for t in range(49):
        col = 49 + t
        u_4x4, v_4x4, w_4x4 = rank_4x4_terms[t]
        
        for idx in range(16):
            U[idx][col] = u_4x4[idx]
        
        # B is 4x8, so B[j, 4+k] -> V[j*8 + 4 + k]
        for j in range(4):
            for k in range(4):
                V[j * 8 + 4 + k][col] = v_4x4[j * 4 + k]
        
        # C is 4x8, so C[k, 4+i] -> W[(4+k)*4 + i]
        for k in range(4):
            for i in range(4):
                W[(4 + k) * 4 + i][col] = w_4x4[k * 4 + i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
