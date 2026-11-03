"""Rewritten program: Strassen-based rank 49 for 4x4 block + schoolbook rank 32 for 4x2 block."""
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We'll build U, V, W as dense integer matrices.
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (24, rank)
    
    # Part 1: 4x4 * 4x4 -> 4x4 using Strassen (rank 49)
    # Part 2: 4x4 * 4x2 -> 4x2 using schoolbook (rank 32)
    # Total rank = 49 + 32 = 81
    
    total_rank = 81
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # --- Part 1: Strassen for 4x4 * 4x4 ---
    # We decompose 4x4 matrices into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = A * B, each Ci is 2x2
    #
    # Strassen 7 products of 2x2 matrices, each using rank 7.
    # Total: 7 * 7 = 49 terms.
    
    # Strassen 2x2 formulas (for scalar 2x2 blocks):
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix product, which itself has rank 7.
    # For each of the 7 Strassen products, we have 7 sub-terms.
    
    # Let's index the 49 terms as (s, t) where s in 0..6 (Strassen product), t in 0..6 (2x2 sub-term)
    
    # First, define the 2x2 Strassen decomposition for scalar entries.
    # For 2x2 matrices P (2x2) and Q (2x2), product R = P*Q.
    # Using Strassen:
    # S1 = (P00+P11)(Q00+Q11)
    # S2 = (P10+P11)Q00
    # S3 = P00(Q01-Q11)
    # S4 = P11(Q10-Q00)
    # S5 = (P00+P01)Q11
    # S6 = (P10-P00)(Q00+Q01)
    # S7 = (P01-P11)(Q10+Q11)
    # R00 = S1+S4-S5+S7
    # R01 = S3+S5
    # R10 = S2+S4
    # R11 = S1-S2+S3+S6
    
    # Now, for each Strassen product s, we need to form the 2x2 matrices P_s and Q_s
    # from the 4x4 blocks, then apply the 2x2 Strassen to get the 2x2 result.
    
    # Let's define the 7 Strassen combinations at the 4x4 level:
    # Each combination produces a 2x2 matrix P_s (from A) and Q_s (from B).
    
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    
    # The 7 products:
    # s=0: P = A00+A11, Q = B00+B11
    # s=1: P = A10+A11, Q = B00
    # s=2: P = A00, Q = B01-B11
    # s=3: P = A11, Q = B10-B00
    # s=4: P = A00+A01, Q = B11
    # s=5: P = A10-A00, Q = B00+B01
    # s=6: P = A01-A11, Q = B10+B11
    
    # For each s, we have a 2x2 matrix P_s and Q_s. Then we compute P_s * Q_s using 2x2 Strassen (7 terms).
    # Each of those 7 terms is a scalar product of entries from P_s and Q_s.
    
    # Let's build the U, V, W entries for these 49 terms.
    
    # Helper: for a 2x2 matrix block in A (which is part of the 4x4 A), 
    # the entries are at positions:
    # A00: rows 0,1; cols 0,1 -> indices 0,1,4,5 in A (flattened as i*m+j)
    # A01: rows 0,1; cols 2,3 -> indices 2,3,6,7
    # A10: rows 2,3; cols 0,1 -> indices 8,9,12,13
    # A11: rows 2,3; cols 2,3 -> indices 10,11,14,15
    
    # For B (4x6), the first 4 columns form the 4x4 part:
    # B00: rows 0,1; cols 0,1 -> indices 0,1,4,5 in B (flattened as j*p+k)
    # B01: rows 0,1; cols 2,3 -> indices 2,3,6,7
    # B10: rows 2,3; cols 0,1 -> indices 8,9,12,13
    # B11: rows 2,3; cols 2,3 -> indices 10,11,14,15
    
    # For C (4x6), the first 4 columns form the 4x4 output:
    # C00: rows 0,1; cols 0,1 -> indices in C: k*n+i, for k in 0,1 and i in 0,1
    # C01: rows 0,1; cols 2,3
    # C10: rows 2,3; cols 0,1
    # C11: rows 2,3; cols 2,3
    
    # Let's define the 2x2 Strassen sub-terms for a single 2x2 * 2x2 product.
    # For P (2x2) and Q (2x2), the 7 scalar products are:
    # t=0: (P00+P11)*(Q00+Q11)
    # t=1: (P10+P11)*Q00
    # t=2: P00*(Q01-Q11)
    # t=3: P11*(Q10-Q00)
    # t=4: (P00+P01)*Q11
    # t=5: (P10-P00)*(Q00+Q01)
    # t=6: (P01-P11)*(Q10+Q11)
    
    # And the output C entries:
    # C00 = t0+t3-t4+t6
    # C01 = t2+t4
    # C10 = t1+t3
    # C11 = t0-t1+t2+t5
    
    # Now, for each Strassen product s (0..6), we have P_s and Q_s which are 2x2 matrices
    # formed from linear combinations of the 4x4 blocks.
    
    # Let's define the linear combinations for P_s and Q_s.
    # Each P_s and Q_s is a 2x2 matrix, with entries that are linear combinations of A and B entries.
    
    # For each s, define:
    # P_s = [[p00, p01], [p10, p11]] where each p is a linear combo of A entries
    # Q_s = [[q00, q01], [q10, q11]] where each q is a linear combo of B entries
    
    # s=0: P = A00+A11, Q = B00+B11
    #   p00 = A00[0,0] + A11[0,0], p01 = A00[0,1] + A11[0,1], etc.
    # s=1: P = A10+A11, Q = B00
    # s=2: P = A00, Q = B01-B11
    # s=3: P = A11, Q = B10-B00
    # s=4: P = A00+A01, Q = B11
    # s=5: P = A10-A00, Q = B00+B01
    # s=6: P = A01-A11, Q = B10+B11
    
    # Let's define the coefficients for each p_ij and q_ij in terms of the block entries.
    
    # A block entries (flattened indices in A, which is 4x4, index = i*4+j):
    # A00: [[0,1],[4,5]]
    # A01: [[2,3],[6,7]]
    # A10: [[8,9],[12,13]]
    # A11: [[10,11],[14,15]]
    
    # B block entries (flattened indices in B, which is 4x6, index = j*6+k, but for first 4 cols):
    # B00: [[0,1],[4,5]]
    # B01: [[2,3],[6,7]]
    # B10: [[8,9],[12,13]]
    # B11: [[10,11],[14,15]]
    
    # For each s and each sub-term t, we need to determine:
    # - Which A entries are involved (with coefficients)
    # - Which B entries are involved (with coefficients)
    # - Which C entries are produced (with coefficients)
    
    # Let's build this systematically.
    
    # First, define the 2x2 block indices for A, B, C.
    # For a 2x2 block at position (br, bc) in the 4x4 matrix:
    # A: rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    #   flattened index: (2*br + dr)*4 + (2*bc + dc) for dr,dc in 0,1
    
    # Let's define the 7 Strassen products and their P, Q matrices.
    
    # For each s, P_s is a 2x2 matrix. Let's define P_s[dr][dc] as a linear combination of A entries.
    # Similarly for Q_s.
    
    # I'll define a function to get the A entry index for block (br, bc) at position (dr, dc):
    def a_idx(br, bc, dr, dc):
        return (2*br + dr) * 4 + (2*bc + dc)
    
    def b_idx(br, bc, dr, dc):
        # B is 4x6, but we're using the first 4 columns
        return (2*br + dr) * 6 + (2*bc + dc)
    
    def c_idx(br, bc, dr, dc):
        # C is 4x6, output is 4x4 for this part
        # C index: k*n + i, where k is col, i is row
        # For block (br, bc): rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
        return (2*bc + dc) * 4 + (2*br + dr)
    
    # Now define the 7 Strassen products:
    # s, P_coeffs, Q_coeffs
    # P_coeffs[dr][dc] = list of (a_idx, coeff)
    # Q_coeffs[dr][dc] = list of (b_idx, coeff)
    
    strassen_products = []
    
    # s=0: P = A00+A11, Q = B00+B11
    P0 = {}
    Q0 = {}
    for dr in range(2):
        for dc in range(2):
            P0[(dr,dc)] = [(a_idx(0,0,dr,dc), 1), (a_idx(1,1,dr,dc), 1)]
            Q0[(dr,dc)] = [(b_idx(0,0,dr,dc), 1), (b_idx(1,1,dr,dc), 1)]
    strassen_products.append((P0, Q0))
    
    # s=1: P = A10+A11, Q = B00
    P1 = {}
    Q1 = {}
    for dr in range(2):
        for dc in range(2):
            P1[(dr,dc)] = [(a_idx(1,0,dr,dc), 1), (a_idx(1,1,dr,dc), 1)]
            Q1[(dr,dc)] = [(b_idx(0,0,dr,dc), 1)]
    strassen_products.append((P1, Q1))
    
    # s=2: P = A00, Q = B01-B11
    P2 = {}
    Q2 = {}
    for dr in range(2):
        for dc in range(2):
            P2[(dr,dc)] = [(a_idx(0,0,dr,dc), 1)]
            Q2[(dr,dc)] = [(b_idx(0,1,dr,dc), 1), (b_idx(1,1,dr,dc), -1)]
    strassen_products.append((P2, Q2))
    
    # s=3: P = A11, Q = B10-B00
    P3 = {}
    Q3 = {}
    for dr in range(2):
        for dc in range(2):
            P3[(dr,dc)] = [(a_idx(1,1,dr,dc), 1)]
            Q3[(dr,dc)] = [(b_idx(1,0,dr,dc), 1), (b_idx(0,0,dr,dc), -1)]
    strassen_products.append((P3, Q3))
    
    # s=4: P = A00+A01, Q = B11
    P4 = {}
    Q4 = {}
    for dr in range(2):
        for dc in range(2):
            P4[(dr,dc)] = [(a_idx(0,0,dr,dc), 1), (a_idx(0,1,dr,dc), 1)]
            Q4[(dr,dc)] = [(b_idx(1,1,dr,dc), 1)]
    strassen_products.append((P4, Q4))
    
    # s=5: P = A10-A00, Q = B00+B01
    P5 = {}
    Q5 = {}
    for dr in range(2):
        for dc in range(2):
            P5[(dr,dc)] = [(a_idx(1,0,dr,dc), 1), (a_idx(0,0,dr,dc), -1)]
            Q5[(dr,dc)] = [(b_idx(0,0,dr,dc), 1), (b_idx(0,1,dr,dc), 1)]
    strassen_products.append((P5, Q5))
    
    # s=6: P = A01-A11, Q = B10+B11
    P6 = {}
    Q6 = {}
    for dr in range(2):
        for dc in range(2):
            P6[(dr,dc)] = [(a_idx(0,1,dr,dc), 1), (a_idx(1,1,dr,dc), -1)]
            Q6[(dr,dc)] = [(b_idx(1,0,dr,dc), 1), (b_idx(1,1,dr,dc), 1)]
    strassen_products.append((P6, Q6))
    
    # Now, for each s and each t (0..6), we have a scalar product term.
    # The 2x2 Strassen sub-terms:
    # t=0: (P00+P11)*(Q00+Q11)
    # t=1: (P10+P11)*Q00
    # t=2: P00*(Q01-Q11)
    # t=3: P11*(Q10-Q00)
    # t=4: (P00+P01)*Q11
    # t=5: (P10-P00)*(Q00+Q01)
    # t=6: (P01-P11)*(Q10+Q11)
    
    # For each t, define which P entries and Q entries are involved.
    # t_coeffs: list of (dr, dc, coeff) for P, and (dr, dc, coeff) for Q
    
    t_defs = [
        # t=0
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # t=1
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # t=2
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),
        # t=3
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),
        # t=4
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # t=5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # t=6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]
    
    # For each (s, t), we compute the U, V, W contributions.
    # The term is: sum over P entries * sum over Q entries
    # U entry: for each A entry in P, add coefficient
    # V entry: for each B entry in Q, add coefficient
    # W entry: for each C entry produced, add coefficient
    
    # The C output for the 2x2 Strassen:
    # C00 = t0+t3-t4+t6
    # C01 = t2+t4
    # C10 = t1+t3
    # C11 = t0-t1+t2+t5
    
    # For each t, the C entries are:
    c_contribs = [
        # t=0: C00 += 1, C11 += 1
        [(0,0,1), (1,1,1)],
        # t=1: C10 += 1, C11 -= 1
        [(1,0,1), (1,1,-1)],
        # t=2: C01 += 1, C11 += 1
        [(0,1,1), (1,1,1)],
        # t=3: C00 += 1, C10 += 1
        [(0,0,1), (1,0,1)],
        # t=4: C00 -= 1, C01 += 1
        [(0,0,-1), (0,1,1)],
        # t=5: C11 += 1
        [(1,1,1)],
        # t=6: C00 += 1
        [(0,0,1)],
    ]
    
    # Now build the 49 terms.
    for s in range(7):
        P_s, Q_s = strassen_products[s]
        for t in range(7):
            term_idx = s * 7 + t
            p_entries, q_entries = t_defs[t]
            
            # For each P entry (dr, dc, coeff_p) and each Q entry (dr2, dc2, coeff_q),
            # the product term contributes to U, V, W.
            
            # Actually, the term is: (sum of P entries) * (sum of Q entries)
            # So U gets the sum of P entries, V gets the sum of Q entries, W gets the C contribution.
            
            # U: for each (dr, dc, coeff_p) in p_entries, add coeff_p to U[a_idx, term_idx]
            for dr, dc, coeff_p in p_entries:
                for a_idx_i, coeff_a in P_s[(dr, dc)]:
                    U[a_idx_i, term_idx] += coeff_p * coeff_a
            
            # V: for each (dr, dc, coeff_q) in q_entries, add coeff_q to V[b_idx, term_idx]
            for dr, dc, coeff_q in q_entries:
                for b_idx_i, coeff_b in Q_s[(dr, dc)]:
                    V[b_idx_i, term_idx] += coeff_q * coeff_b
            
            # W: for each C entry (br, bc, coeff_c) in c_contribs[t], add coeff_c to W[c_idx, term_idx]
            for br, bc, coeff_c in c_contribs[t]:
                for dr in range(2):
                    for dc in range(2):
                        c_idx_i = c_idx(br, bc, dr, dc)
                        W[c_idx_i, term_idx] += coeff_c
    
    # --- Part 2: Schoolbook for 4x4 * 4x2 ---
    # The last 2 columns of B (cols 4,5) and C (cols 4,5).
    # For each i in 0..3, j in 0..3, k in 4..5:
    #   U[i*4+j, term] = 1
    #   V[j*6+k, term] = 1
    #   W[k*4+i, term] = 1
    
    schoolbook_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                term_idx = schoolbook_offset + (i*4 + j)*2 + (k-4)
                U[i*4+j, term_idx] = 1
                V[j*6+k, term_idx] = 1
                W[k*4+i, term_idx] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
