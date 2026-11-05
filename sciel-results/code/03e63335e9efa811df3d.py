import time

_START_TIME = time.time()
TIME_LIMIT = 150.0

def construct():
    """
    Returns a rank-65 exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    Decomposition: M_{4,4,5} = M_{4,4,4} \oplus M_{4,4,1}.
    Rank(M_{4,4,4}) = 49 (Strassen's algorithm).
    Rank(M_{4,4,1}) = 16 (Schoolbook).
    Total Rank = 65.
    """
    n, m, p = 4, 4, 5
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 parameters
    # Each term: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict or list of (row, col, coeff) for A
    # b_coeffs: dict or list of (row, col, coeff) for B
    # c_coeffs: dict or list of (row, col, coeff) for C
    
    # Define the 7 Strassen terms for 2x2x2
    # M0 = (A00+A11)(B00+B11) -> C00+=M0, C11+=M0
    # M1 = (A10+A11)B00       -> C10+=M1, C11-=M1
    # M2 = A00(B01-B11)       -> C01+=M2, C11+=M2
    # M3 = A11(B10-B00)       -> C00+=M3, C10+=M3
    # M4 = (A00+A01)B11       -> C00-=M4, C01+=M4
    # M5 = (A10-A00)(B00+B01) -> C11+=M5
    # M6 = (A01-A11)(B10+B11) -> C00+=M6
    
    s2x2 = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)],           [(1,0,1), (1,1,-1)]),
        ([(0,0,1)],           [(0,1,1), (1,1,-1)],[(0,1,1), (1,1,1)]),
        ([(1,1,1)],           [(1,0,1), (0,0,-1)],[(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)],           [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)],[(0,0,1), (0,1,1)],  [(1,1,1)]),
        ([(0,1,1), (1,1,-1)],[(1,0,1), (1,1,1)],  [(0,0,1)]),
    ]
    
    # Level 1: 4x4x4 decomposition using 2x2 blocks
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    # C blocks: C00 (0,0), C01 (0,1), C10 (1,0), C11 (1,1)
    
    # The same Strassen structure applies to the blocks
    l1_forms = s2x2 # The structure is identical
    
    # Helper to map local 2x2 indices to global 4x4 indices
    def map_idx(block, local):
        # block: (br, bc), local: (r, c)
        # global row = 2*br + r
        # global col = 2*bc + c
        return 2*block[0] + local[0], 2*block[1] + local[1]

    # Generate 49 terms for 4x4x4
    for l1_idx in range(7):
        p_blks, q_blks, c_blks = l1_forms[l1_idx]
        for l2_idx in range(7):
            a_mask, b_mask, c_mask = s2x2[l2_idx]
            
            # U contributions
            for (abr, abc, ac) in p_blks:
                for (ar, ac2, acoeff) in a_mask:
                    gr, gc = map_idx((abr, abc), (ar, ac2))
                    U[gr * m + gc][col] += ac * acoeff
            
            # V contributions (only for first 4 columns of B, i.e., k < 4)
            for (qbr, qbc, qc) in q_blks:
                for (qr, qc2, qcoeff) in b_mask:
                    gr, gc = map_idx((qbr, qbc), (qr, qc2))
                    # gc is the column index in B. Since we are in 4x4x4 part, gc is 0..3
                    V[gr * p + gc][col] += qc * qcoeff
            
            # W contributions
            for (cbr, cbc, cc) in c_blks:
                for (cr, cc2, wcoeff) in c_mask:
                    gi, gj = map_idx((cbr, cbc), (cr, cc2))
                    # gi is row of C, gj is col of C
                    # W index: k*n + i, where k is col of C, i is row of C
                    W[gj * n + gi][col] += cc * wcoeff
            
            col += 1
    
    # Schoolbook for 4x4 * 4x1 (column 4 of B)
    # C_i4 = sum_j A_ij B_j4
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
