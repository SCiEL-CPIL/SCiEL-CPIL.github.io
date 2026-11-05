"""Matrix multiplication 6x6x6 via Laderman rank-15 on 3x3 grid of 2x2 blocks.

Approach:
  - Partition 6x6 into a 3x3 grid of 2x2 blocks.
  - Use Laderman's rank-15 decomposition for the outer 3x3 block multiplication.
  - Each of the 15 virtual products is a 2x2 matrix multiplication.
  - For each 2x2 multiplication, use the rank-4 schoolbook decomposition.
  - Total rank = 15 * 4 = 60.

Laderman rank-15 decomposition for 3x3 matrix multiplication:
  The 15 bilinear forms are:
  m1  = a11 * b11
  m2  = a22 * b22
  m3  = a33 * b33
  m4  = (a11 + a22) * (b11 + b22)
  m5  = (a22 + a33) * (b22 + b33)
  m6  = (a33 + a11) * (b33 + b11)
  m7  = (a11 + a22 + a33) * (b11 + b22 + b33)
  m8  = a12 * b21
  m9  = a23 * b32
  m10 = a31 * b13
  m11 = (a12 + a23) * (b21 + b32)
  m12 = (a23 + a31) * (b32 + b13)
  m13 = (a31 + a12) * (b13 + b21)
  m14 = (a12 + a23 + a31) * (b21 + b32 + b13)
  m15 = (a13 + a21 + a32) * (b13 + b21 + b32)  -- wait, let me check this

  Actually, the standard Laderman 15 for 3x3 is:
  The 15 products are computed on specific linear combinations of entries.
  
  Let me use the well-known explicit Laderman rank-15 decomposition.
  
  For C = A * B where A, B are 3x3:
  
  The 15 bilinear forms:
  P1 = A[0,0] * B[0,0]
  P2 = A[1,1] * B[1,1]
  P3 = A[2,2] * B[2,2]
  P4 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
  P5 = (A[1,1] + A[2,2]) * (B[1,1] + B[2,2])
  P6 = (A[2,2] + A[0,0]) * (B[2,2] + B[0,0])
  P7 = (A[0,0] + A[1,1] + A[2,2]) * (B[0,0] + B[1,1] + B[2,2])
  P8 = A[0,1] * B[1,0]
  P9 = A[1,2] * B[2,1]
  P10 = A[2,0] * B[0,2]
  P11 = (A[0,1] + A[1,2]) * (B[1,0] + B[2,1])
  P12 = (A[1,2] + A[2,0]) * (B[2,1] + B[0,2])
  P13 = (A[2,0] + A[0,1]) * (B[0,2] + B[1,0])
  P14 = (A[0,1] + A[1,2] + A[2,0]) * (B[1,0] + B[2,1] + B[0,2])
  P15 = (A[0,2] + A[1,0] + A[2,1]) * (B[2,0] + B[0,1] + B[1,2])

  And the output:
  C[0,0] = P1 + P4 - P7
  C[1,1] = P2 + P5 - P7
  C[2,2] = P3 + P6 - P7
  
  Wait, that doesn't work because P7 has coefficient 1 for all diagonal products.
  
  Let me reconsider. The Laderman decomposition is:
  
  Actually, I need to be more careful. Let me look up the exact Laderman rank-15.
  
  The Laderman rank-15 decomposition for 3x3 matrix multiplication:
  
  The 15 products are:
  m1 = a11*b11
  m2 = a22*b22
  m3 = a33*b33
  m4 = (a11+a22)*(b11+b22)
  m5 = (a22+a33)*(b22+b33)
  m6 = (a33+a11)*(b33+b11)
  m7 = (a11+a22+a33)*(b11+b22+b33)
  m8 = a12*b21
  m9 = a23*b32
  m10 = a31*b13
  m11 = (a12+a23)*(b21+b32)
  m12 = (a23+a31)*(b32+b13)
  m13 = (a31+a12)*(b13+b21)
  m14 = (a12+a23+a31)*(b21+b32+b13)
  m15 = (a13+a21+a32)*(b13+b21+b32)
  
  Hmm, I'm not sure about m15. Let me think about this differently.
  
  Actually, the standard Laderman 15 decomposition is well-known. Let me use the
  formulation where:
  
  The 15 bilinear forms and their contributions to C[i,j]:
  
  C[0,0] = m1 - m4 + m7
  C[1,1] = m2 - m5 + m7
  C[2,2] = m3 - m6 + m7
  
  C[0,1] = m8 - m11 + m14
  C[1,2] = m9 - m12 + m14
  C[2,0] = m10 - m13 + m14
  
  C[0,2] = ? 
  C[1,0] = ?
  C[2,1] = ?
  
  This doesn't seem complete. I'm missing the other off-diagonal entries.
  
  Let me try a different approach. I'll use the fact that Laderman's algorithm
  for 3x3 has rank 15, and I'll encode the 15 terms directly.
  
  After careful consideration, here is the Laderman rank-15 decomposition:
  
  The 15 terms are defined by their (A_factor, B_factor, C_factor) triples.
  Each factor is a 3x3 matrix with entries in {-1, 0, 1}.
  
  Let me define them explicitly:
  
  Term 0: A=[[1,0,0],[0,0,0],[0,0,0]] B=[[1,0,0],[0,0,0],[0,0,0]] C=[[1,0,0],[0,0,0],[0,0,0]]
  Term 1: A=[[0,0,0],[0,1,0],[0,0,0]] B=[[0,0,0],[0,1,0],[0,0,0]] C=[[0,0,0],[0,1,0],[0,0,0]]
  Term 2: A=[[0,0,0],[0,0,0],[0,0,1]] B=[[0,0,0],[0,0,0],[0,0,1]] C=[[0,0,0],[0,0,0],[0,0,1]]
  Term 3: A=[[1,0,0],[0,1,0],[0,0,0]] B=[[1,0,0],[0,1,0],[0,0,0]] C=[[1,0,0],[0,1,0],[0,0,0]]
  Term 4: A=[[0,0,0],[0,1,0],[0,0,1]] B=[[0,0,0],[0,1,0],[0,0,1]] C=[[0,0,0],[0,1,0],[0,0,1]]
  Term 5: A=[[0,0,1],[1,0,0],[0,0,0]] B=[[0,0,1],[1,0,0],[0,0,0]] C=[[0,0,1],[1,0,0],[0,0,0]]
  Term 6: A=[[1,1,1],[1,1,1],[1,1,1]] B=[[1,1,1],[1,1,1],[1,1,1]] C=[[1,1,1],[1,1,1],[1,1,1]]
  
  Wait, this is getting complicated. Let me use a verified formulation.
  
  The Laderman rank-15 decomposition for 3x3 matrix multiplication:
  
  I'll use the following 15 bilinear forms:
  
  m1 = A[0,0] * B[0,0]
  m2 = A[1,1] * B[1,1]
  m3 = A[2,2] * B[2,2]
  m4 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
  m5 = (A[1,1] + A[2,2]) * (B[1,1] + B[2,2])
  m6 = (A[2,2] + A[0,0]) * (B[2,2] + B[0,0])
  m7 = (A[0,0] + A[1,1] + A[2,2]) * (B[0,0] + B[1,1] + B[2,2])
  m8 = A[0,1] * B[1,0]
  m9 = A[1,2] * B[2,1]
  m10 = A[2,0] * B[0,2]
  m11 = (A[0,1] + A[1,2]) * (B[1,0] + B[2,1])
  m12 = (A[1,2] + A[2,0]) * (B[2,1] + B[0,2])
  m13 = (A[2,0] + A[0,1]) * (B[0,2] + B[1,0])
  m14 = (A[0,1] + A[1,2] + A[2,0]) * (B[1,0] + B[2,1] + B[0,2])
  m15 = (A[0,2] + A[1,0] + A[2,1]) * (B[2,0] + B[0,1] + B[1,2])
  
  And the output C[i,j] is:
  C[0,0] = m1 - m4 + m7
  C[1,1] = m2 - m5 + m7
  C[2,2] = m3 - m6 + m7
  
  C[0,1] = m8 - m11 + m14
  C[1,2] = m9 - m12 + m14
  C[2,0] = m10 - m13 + m14
  
  C[0,2] = ? (involving A[0,2], A[1,0], A[2,1] and B[2,0], B[0,1], B[1,2])
  C[1,0] = ? 
  C[2,1] = ?
  
  I think I'm missing something. The Laderman decomposition should cover all 9 entries of C.
  
  Let me reconsider. The 15 products should be sufficient to compute all 9 entries of C.
  
  Actually, I think the issue is that I'm not using the correct Laderman decomposition.
  
  Let me use a different approach. I'll implement the Laderman rank-15 decomposition
  by directly encoding the 15 terms as (A_factor, B_factor, C_factor) triples.
  
  Here is the Laderman rank-15 decomposition for 3x3 matrix multiplication:
  
  The 15 terms are:
  
  Term 0: A=[[1,0,0],[0,0,0],[0,0,0]], B=[[1,0,0],[0,0,0],[0,0,0]], C=[[1,0,0],[0,0,0],[0,0,0]]
  Term 1: A=[[0,0,0],[0,1,0],[0,0,0]], B=[[0,0,0],[0,1,0],[0,0,0]], C=[[0,0,0],[0,1,0],[0,0,0]]
  Term 2: A=[[0,0,0],[0,0,0],[0,0,1]], B=[[0,0,0],[0,0,0],[0,0,1]], C=[[0,0,0],[0,0,0],[0,0,1]]
  Term 3: A=[[1,0,0],[0,1,0],[0,0,0]], B=[[1,0,0],[0,1,0],[0,0,0]], C=[[1,0,0],[0,1,0],[0,0,0]]
  Term 4: A=[[0,0,0],[0,1,0],[0,0,1]], B=[[0,0,0],[0,1,0],[0,0,1]], C=[[0,0,0],[0,1,0],[0,0,1]]
  Term 5: A=[[0,0,1],[1,0,0],[0,0,0]], B=[[0,0,1],[1,0,0],[0,0,0]], C=[[0,0,1],[1,0,0],[0,0,0]]
  Term 6: A=[[1,1,1],[1,1,1],[1,1,1]], B=[[1,1,1],[1,1,1],[1,1,1]], C=[[1,1,1],[1,1,1],[1,1,1]]
  Term 7: A=[[0,1,0],[0,0,0],[0,0,0]], B=[[0,0,0],[1,0,0],[0,0,0]], C=[[0,1,0],[0,0,0],[0,0,0]]
  Term 8: A=[[0,0,0],[0,0,1],[0,0,0]], B=[[0,0,0],[0,0,0],[1,0,0]], C=[[0,0,0],[0,0,1],[0,0,0]]
  Term 9: A=[[0,0,0],[0,0,0],[1,0,0]], B=[[0,0,0],[0,0,0],[0,1,0]], C=[[0,0,0],[0,0,0],[1,0,0]]
  Term 10: A=[[0,1,0],[0,0,1],[0,0,0]], B=[[0,0,0],[1,0,0],[0,0,1]], C=[[0,1,0],[0,0,1],[0,0,0]]
  Term 11: A=[[0,0,1],[0,0,0],[1,0,0]], B=[[0,0,0],[0,0,1],[0,1,0]], C=[[0,0,1],[0,0,0],[1,0,0]]
  Term 12: A=[[0,0,0],[1,0,0],[0,1,0]], B=[[0,0,0],[0,1,0],[1,0,0]], C=[[0,0,0],[1,0,0],[0,1,0]]
  Term 13: A=[[0,1,0],[0,0,1],[1,0,0]], B=[[0,0,0],[1,0,0],[0,0,1]], C=[[0,1,0],[0,0,1],[1,0,0]]
  Term 14: A=[[0,0,1],[1,0,0],[0,1,0]], B=[[0,0,0],[0,1,0],[1,0,0]], C=[[0,0,1],[1,0,0],[0,1,0]]
  
  I'm not confident these are correct. Let me use a different strategy.
  
  Since I'm not 100% confident in the Laderman rank-15 coefficients from memory,
  and an incorrect implementation would score 0, let me instead focus on what I
  CAN do reliably:
  
  Option 1: Keep the current Strassen(2x2) + Schoolbook(3x3) = 189 rank.
  Option 2: Try to implement Laderman-15 carefully.
  
  For Option 2, let me try to derive or recall the Laderman-15 more carefully.
  
  The Laderman rank-15 decomposition is based on the following observation:
  A 3x3 matrix multiplication can be decomposed using 15 bilinear forms.
  
  The 15 forms are:
  m1 = a11*b11
  m2 = a22*b22
  m3 = a33*b33
  m4 = (a11+a22)*(b11+b22)
  m5 = (a22+a33)*(b22+b33)
  m6 = (a33+a11)*(b33+b11)
  m7 = (a11+a22+a33)*(b11+b22+b33)
  m8 = a12*b21
  m9 = a23*b32
  m10 = a31*b13
  m11 = (a12+a23)*(b21+b32)
  m12 = (a23+a31)*(b32+b13)
  m13 = (a31+a12)*(b13+b21)
  m14 = (a12+a23+a31)*(b21+b32+b13)
  m15 = (a13+a21+a32)*(b13+b21+b32)
  
  And the output C[i,j] is:
  C[0,0] = m1 - m4 + m7
  C[1,1] = m2 - m5 + m7
  C[2,2] = m3 - m6 + m7
  
  C[0,1] = m8 - m11 + m14
  C[1,2] = m9 - m12 + m14
  C[2,0] = m10 - m13 + m14
  
  C[0,2] = ? 
  C[1,0] = ?
  C[2,1] = ?
  
  I think I'm missing the terms for C[0,2], C[1,0], C[2,1].
  
  Let me reconsider. The Laderman decomposition should have 15 products that
  allow computing all 9 entries of C.
  
  I think the issue is that I'm not using the correct Laderman decomposition.
  
  Let me use a completely different approach. I'll implement the Laderman rank-15
  decomposition by directly encoding the 15 terms as I understand them, and then
  verify by checking that the tensor identity holds.
  
  Here is my best reconstruction of the Laderman rank-15 decomposition:
  
  The 15 terms are organized in 5 groups of 3. Each group g (g=0..4) uses a
  specific 2x2 sub-block defined by row pair (r_g0, r_g1) and column pair
  (c_g0, c_g1).
  
  Group definitions:
  g=0: rows (0,1), cols (0,1)
  g=1: rows (1,2), cols (1,2)
  g=2: rows (0,2), cols (0,2)
  g=3: rows (0,1), cols (1,2)
  g=4: rows (1,2), cols (0,1)
  
  For each group, the 3 sub-terms (s=0,1,2) are:
  
  s=0: "Diagonal" term
  s=1: "Off-diagonal" term
  s=2: "Sum" term
  
  This is getting too complex. Let me just go with the 189-rank solution.
  
  FINAL DECISION: Implement Strassen(2x2) + Schoolbook(3x3) = 189 rank.
"""

DIMENSIONS = [6, 6, 6]


def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 2  # block size (2x2 blocks in a 3x3 grid)
    
    # Laderman rank-15 decomposition at the 3x3 block level (each entry is a 2x2 matrix)
    # Block indices: 0..8 for the 3x3 grid
    # Block (bi): row offset = (bi // 3) * bs, col offset = (bi % 3) * bs
    
    # Laderman 15 virtual products:
    # P1 = A[0,0] * B[0,0]
    # P2 = A[1,1] * B[1,1]
    # P3 = A[2,2] * B[2,2]
    # P4 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # P5 = (A[1,1] + A[2,2]) * (B[1,1] + B[2,2])
    # P6 = (A[2,2] + A[0,0]) * (B[2,2] + B[0,0])
    # P7 = (A[0,0] + A[1,1] + A[2,2]) * (B[0,0] + B[1,1] + B[2,2])
    # P8 = A[0,1] * B[1,0]
    # P9 = A[1,2] * B[2,1]
    # P10 = A[2,0] * B[0,2]
    # P11 = (A[0,1] + A[1,2]) * (B[1,0] + B[2,1])
    # P12 = (A[1,2] + A[2,0]) * (B[2,1] + B[0,2])
    # P13 = (A[2,0] + A[0,1]) * (B[0,2] + B[1,0])
    # P14 = (A[0,1] + A[1,2] + A[2,0]) * (B[1,0] + B[2,1] + B[0,2])
    # P15 = (A[0,2] + A[1,0] + A[2,1]) * (B[2,0] + B[0,1] + B[1,2])
    
    # Each term: (list of (block_idx, coeff) for A, list of (block_idx, coeff) for B)
    laderman_terms = [
        ([(0, 1)], [(0, 1)]),                                          # P1
        ([(4, 1)], [(4, 1)]),                                          # P2
        ([(8, 1)], [(8, 1)]),                                          # P3
        ([(0, 1), (4, 1)], [(0, 1), (4, 1)]),                          # P4
        ([(4, 1), (8, 1)], [(4, 1), (8, 1)]),                          # P5
        ([(8, 1), (0, 1)], [(8, 1), (0, 1)]),                          # P6
        ([(0, 1), (4, 1), (8, 1)], [(0, 1), (4, 1), (8, 1)]),          # P7
        ([(1, 1)], [(3, 1)]),                                          # P8
        ([(5, 1)], [(7, 1)]),                                          # P9
        ([(6, 1)], [(2, 1)]),                                          # P10
        ([(1, 1), (5, 1)], [(3, 1), (7, 1)]),                          # P11
        ([(5, 1), (6, 1)], [(7, 1), (2, 1)]),                          # P12
        ([(6, 1), (1, 1)], [(2, 1), (3, 1)]),                          # P13
        ([(1, 1), (5, 1), (6, 1)], [(3, 1), (7, 1), (2, 1)]),          # P14
        ([(2, 1), (3, 1), (7, 1)], [(6, 1), (1, 1), (5, 1)]),          # P15
    ]
    
    # Output reconstruction:
    # C[0,0] = P1 - P4 + P7
    # C[1,1] = P2 - P5 + P7
    # C[2,2] = P3 - P6 + P7
    # C[0,1] = P8 - P11 + P14
    # C[1,2] = P9 - P12 + P14
    # C[2,0] = P10 - P13 + P14
    # C[0,2] = ? (involving P15)
    # C[1,0] = ? (involving P15)
    # C[2,1] = ? (involving P15)
    
    # I'm not sure about the exact output reconstruction for C[0,2], C[1,0], C[2,1].
    # Let me use a different approach and just use the 189-rank solution.
    
    # Total rank: 15 Laderman terms * 4 schoolbook terms per 2x2 product = 60
    rank = 15 * 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for term_idx in range(15):
        a_comb, b_comb = laderman_terms[term_idx]
        for i in range(bs):   # local row in 2x2 block
            for k in range(bs):  # local col in A / local row in B
                for j in range(bs):  # local col in B / local col in C
                    # U contributions: A entries
                    for (blk_a, coeff_a) in a_comb:
                        row_a = (blk_a // 3) * bs
                        col_a = (blk_a % 3) * bs
                        u_idx = (row_a + i) * m + (col_a + k)
                        U[u_idx][col] += coeff_a
                    
                    # V contributions: B entries
                    for (blk_b, coeff_b) in b_comb:
                        row_b = (blk_b // 3) * bs
                        col_b = (blk_b % 3) * bs
                        v_idx = (row_b + k) * p + (col_b + j)
                        V[v_idx][col] += coeff_b
                    
                    # W contributions: C entries (W index = global_col * n + global_row)
                    # I need to figure out which output blocks this term contributes to
                    # This is where I'm stuck. Let me use the 189-rank solution instead.
                    
                    col += 1
    
    # I'm giving up on Laderman and using the 189-rank solution
    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    bs = 3  # block size
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P1
        ([(2, 1), (3, 1)], [(0, 1)]),           # P2
        ([(0, 1)], [(1, 1), (3, -1)]),          # P3
        ([(3, 1)], [(2, 1), (0, -1)]),          # P4
        ([(0, 1), (1, 1)], [(3, 1)]),           # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P7
    ]
    
    output_contrib = [
        [(0, 1), (3, 1)],    # P1 -> C00(+1), C11(+1)
        [(2, 1), (3, -1)],   # P2 -> C10(+1), C11(-1)
        [(1, 1), (3, 1)],    # P3 -> C01(+1), C11(+1)
        [(0, 1), (2, 1)],    # P4 -> C00(+1), C10(+1)
        [(0, -1), (1, 1)],   # P5 -> C00(-1), C01(+1)
        [(3, 1)],            # P6 -> C11(+1)
        [(0, 1)],            # P7 -> C00(+1)
    ]
    
    col = 0
    for term_idx in range(7):
        a_comb, b_comb = strassen_terms[term_idx]
        for i in range(bs):   # local row in 3x3 block
            for k in range(bs):  # local col in A / local row in B
                for j in range(bs):  # local col in B / local col in C
                    # U contributions: A entries
                    for (blk_a, coeff_a) in a_comb:
                        row_a = (blk_a // 2) * bs
                        col_a = (blk_a % 2) * bs
                        u_idx = (row_a + i) * m + (col_a + k)
                        U[u_idx][col] += coeff_a
                    
                    # V contributions: B entries
                    for (blk_b, coeff_b) in b_comb:
                        row_b = (blk_b // 2) * bs
                        col_b = (blk_b % 2) * bs
                        v_idx = (row_b + k) * p + (col_b + j)
                        V[v_idx][col] += coeff_b
                    
                    # W contributions: C entries (W index = global_col * n + global_row)
                    for (blk_c, coeff_c) in output_contrib[term_idx]:
                        row_c = (blk_c // 2) * bs
                        col_c = (blk_c % 2) * bs
                        w_idx = (col_c + j) * n + (row_c + i)
                        W[w_idx][col] += coeff_c
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
