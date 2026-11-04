"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 6x6x6 using Strassen-style block decomposition.
    
    Strategy:
    - Partition 6x6 matrices into 2x2 blocks of 3x3 submatrices.
    - Apply Strassen's 7-term multiplication at the block level.
    - Use schoolbook (27-term) multiplication for each 3x3 block product.
    - Total rank: 7 * 27 = 189 (vs schoolbook 216).
    """
    n, m, p = DIMENSIONS
    bs = 3  # block size
    
    # Strassen formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B10)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # A_blocks[t]: list of (block_row, block_col, coeff) for the left matrix in Strassen term t
    # B_blocks[t]: list of (block_row, block_col, coeff) for the right matrix in Strassen term t
    A_blocks = [
        [(0,0,1), (1,1,1)],   # M1: A00 + A11
        [(1,0,1), (1,1,1)],   # M2: A10 + A11
        [(0,0,1)],            # M3: A00
        [(1,1,1)],            # M4: A11
        [(0,0,1), (0,1,1)],   # M5: A00 + A01
        [(1,0,1), (0,0,-1)],  # M6: A10 - A00
        [(0,1,1), (1,1,-1)],  # M7: A01 - A11
    ]
    B_blocks = [
        [(0,0,1), (1,1,1)],   # M1: B00 + B11
        [(0,0,1)],            # M2: B00
        [(1,0,1), (1,1,-1)],  # M3: B10 - B11
        [(0,1,1), (1,0,-1)],  # M4: B01 - B10
        [(1,1,1)],            # M5: B11
        [(0,0,1), (0,1,1)],   # M6: B00 + B01
        [(1,0,1), (1,1,1)],   # M7: B10 + B11
    ]
    
    # c_contributions[(I,J)]: list of (strassen_term_idx, coeff)
    c_contributions = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],   # M1 + M4 - M5 + M7
        (0,1): [(2,1), (4,1)],                   # M3 + M5
        (1,0): [(1,1), (3,1)],                   # M2 + M4
        (1,1): [(0,1), (1,-1), (2,1), (5,1)],   # M1 - M2 + M3 + M6
    }
    
    rank = 7 * bs * bs * bs  # 189
    U = [[0] * rank for _ in range(n * m)]  # (36, 189)
    V = [[0] * rank for _ in range(m * p)]  # (36, 189)
    W = [[0] * rank for _ in range(p * n)]  # (36, 189)
    
    col = 0
    for t in range(7):
        Ab = A_blocks[t]
        Bb = B_blocks[t]
        
        # For each of the 27 scalar products in the 3x3 * 3x3 multiplication:
        for r in range(bs):
            for s in range(bs):
                for u in range(bs):
                    # X[r,u] * Y[u,s] contributes to output C[r,s] for this block product
                    # X[r,u] = sum over (Ia, Ja, ca) in Ab: ca * A_block[Ia,Ja][r,u]
                    #   A_block[Ia,Ja][r,u] is at matrix position (Ia*bs+r, Ja*bs+u)
                    #   U index = (Ia*bs+r)*m + (Ja*bs+u)
                    
                    # Y[u,s] = sum over (Ib, Jb, cb) in Bb: cb * B_block[Ib,Jb][u,s]
                    #   B_block[Ib,Jb][u,s] is at matrix position (Ib*bs+u, Jb*bs+s)
                    #   V index = (Ib*bs+u)*p + (Jb*bs+s)
                    
                    # Output: this product contributes to C_block[?][r,s]
                    # The Strassen term t contributes to various C blocks with certain coefficients
                    for (Ic, Jc) in [(0,0), (0,1), (1,0), (1,1)]:
                        for (t2, coeff) in c_contributions[(Ic, Jc)]:
                            if t2 == t:
                                # C[Ic, Jc][r,s] gets this contribution
                                # Global C position: row = Ic*bs+r, col = Jc*bs+s
                                # W index = col*n + row = (Jc*bs+s)*n + (Ic*bs+r)
                                w_idx = (Jc * bs + s) * n + (Ic * bs + r)
                                W[w_idx][col] += coeff
                    
                    # Fill U
                    for (Ia, Ja, ca) in Ab:
                        u_idx = (Ia * bs + r) * m + (Ja * bs + u)
                        U[u_idx][col] += ca
                    
                    # Fill V
                    for (Ib, Jb, cb) in Bb:
                        v_idx = (Ib * bs + u) * p + (Jb * bs + s)
                        V[v_idx][col] += cb
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
