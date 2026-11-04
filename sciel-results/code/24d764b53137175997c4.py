"""
Rank-88 exact tensor decomposition for 3x4x8 matrix multiplication.

Uses Strassen's 7-multiplication algorithm for the 2x2x2 sub-problems.
The 3x4x8 product is decomposed into:
  - 4 blocks of type 2x2x4 (top 2 rows of A, each 2-column block of B)
  - 4 blocks of type 1x2x4 (bottom row of A, each 2-column block of B)

Rank of 2x2x4 = 14 (two Strassen 2x2x2 blocks)
Rank of 1x2x4 = 8  (schoolbook: 1*2*4)
Total rank = 4*14 + 4*8 = 56 + 32 = 88
"""

DIMENSIONS = [3, 4, 8]

# Strassen 2x2x2: 7 rank-1 terms
# Indices: A is 2x2 (a00,a01;a10,a11), B is 2x2 (b00,b01;b10,b11)
# C = A*B, C[i,j] = sum_k A[i,k]*B[k,j]
#
# Strassen:
# m1 = (a00+a11)(b00+b11)
# m2 = (a10+a11)b00
# m3 = a00(b01-b10)
# m4 = a11(b10-b00)
# m5 = (a00+a01)b10
# m6 = (a10-a00)(b00+b01)
# m7 = (a01-a11)(b10+b11)
#
# c00 = m1 + m4 - m5 + m7
# c01 = m3 + m5
# c10 = m2 + m4
# c11 = m1 - m2 + m3 + m6

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen.
    U: 4x7 (A indices: a00,a01,a10,a11)
    V: 4x7 (B indices: b00,b10,b01,b11) -- note: B is column-major? 
    Actually for matrix mult tensor, A is (i,k), B is (k,j), C is (i,j).
    U has rows indexed by (i,k) pairs, V by (k,j) pairs, W by (i,j) pairs.
    """
    # A indices: a00=(0,0), a01=(0,1), a10=(1,0), a11=(1,1)
    # B indices: b00=(0,0), b01=(0,1), b10=(1,0), b11=(1,1)
    # C indices: c00=(0,0), c01=(0,1), c10=(1,0), c11=(1,1)
    
    # Strassen mults:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b10)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b10
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    
    U = [[0]*7 for _ in range(4)]  # rows: a00, a01, a10, a11
    V = [[0]*7 for _ in range(4)]  # rows: b00, b01, b10, b11
    W = [[0]*7 for _ in range(4)]  # rows: c00, c01, c10, c11
    
    # m1: (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # c00 += m1, c11 += m1
    # c01, c10: 0
    
    # m2: (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # c10 += m2, c11 -= m2
    
    # m3: a00(b01-b10)
    U[0][2] = 1
    V[1][2] = 1; V[2][2] = -1
    W[1][2] = 1; W[3][2] = 1  # c01 += m3, c11 += m3
    
    # m4: a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # c00 += m4, c10 += m4
    
    # m5: (a00+a01)b10
    U[0][4] = 1; U[1][4] = 1
    V[2][4] = 1
    W[0][4] = -1; W[1][4] = 1  # c00 -= m5, c01 += m5
    
    # m6: (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # c11 += m6
    
    # m7: (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # c00 += m7
    
    return U, V, W

def _block_2x2x4(col_offset):
    """
    Compute 2x2x4 block using two Strassen 2x2x2 blocks.
    A is 2x2, B is 2x4, C is 2x4.
    B columns: (b00, b10) and (b01, b11) for first 2x2x2, then (b02, b12) and (b03, b13) for second.
    
    Returns U (4x14), V (8x14), W (8x14) where:
    U rows: a00, a01, a10, a11
    V rows: b00, b01, b02, b03, b10, b11, b12, b13 (B is 2x4, indexed by [k][j])
    W rows: c00, c01, c02, c03, c10, c11, c12, c13 (C is 2x4)
    
    col_offset: which pair of B columns this handles (0,1 or 2,3)
    """
    U1, V1, W1 = _strassen_2x2x2()
    U2, V2, W2 = _strassen_2x2x2()
    
    rank = 14
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]
    
    # First 2x2x2: columns 0,1 of B (b00,b10 -> col 0; b01,b11 -> col 1)
    # B row 0: b00, b01, b02, b03 -> indices 0,1,2,3
    # B row 1: b10, b11, b12, b13 -> indices 4,5,6,7
    # For first 2x2x2: b00=V[0], b01=V[1], b10=V[4], b11=V[5]
    # Strassen V rows: b00, b01, b10, b11 -> map to global V[0], V[1], V[4], V[5]
    
    for r in range(4):
        for c in range(7):
            U[r][c] = U1[r][c]
        for c in range(7):
            U[r][c+7] = U2[r][c]
    
    # V mapping for first block: V1 rows 0,1,2,3 -> global rows 0,1,4,5
    v1_map = [0, 1, 4, 5]  # b00->0, b01->1, b10->4, b11->5
    for r in range(4):
        for c in range(7):
            V[v1_map[r]][c] = V1[r][c]
    
    # V mapping for second block: V2 rows 0,1,2,3 -> global rows 2,3,6,7
    v2_map = [2, 3, 6, 7]  # b02->2, b03->3, b12->6, b13->7
    for r in range(4):
        for c in range(7):
            V[v2_map[r]][c+7] = V2[r][c]
    
    # W mapping: first block c00,c01 -> global W[0], W[1]; second block c00,c01 -> global W[2], W[3]
    # First block: c00->0, c01->1, c10->4, c11->5
    w1_map = [0, 1, 4, 5]
    for r in range(4):
        for c in range(7):
            W[w1_map[r]][c] = W1[r][c]
    
    # Second block: c00->2, c01->3, c10->6, c11->7
    w2_map = [2, 3, 6, 7]
    for r in range(4):
        for c in range(7):
            W[w2_map[r]][c+7] = W2[r][c]
    
    return U, V, W

def _block_1x2x4():
    """
    Compute 1x2x4 block using schoolbook.
    A is 1x2, B is 2x4, C is 1x4.
    Rank = 1*2*4 = 8.
    
    U: 2x8 (a0, a1)
    V: 8x8 (b00,b01,b02,b03,b10,b11,b12,b13)
    W: 4x8 (c0,c1,c2,c3)
    """
    rank = 8
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(4)]
    
    col = 0
    for k in range(2):
        for j in range(4):
            U[k][col] = 1
            V[k*4+j][col] = 1
            W[j][col] = 1
            col += 1
    
    return U, V, W

def construct():
    """
    Construct rank-88 decomposition for 3x4x8 matrix multiplication.
    
    U: (3*4=12) x rank
    V: (4*8=32) x rank  
    W: (8*3=24) x rank
    
    A is 3x4: rows i=0,1,2; cols j=0,1,2,3
    B is 4x8: rows k=0,1,2,3; cols l=0,...,7
    C is 3x8: rows i=0,1,2; cols l=0,...,7
    
    Decomposition:
    - Top 2 rows of A (i=0,1), split B into 4 column-pairs (l=0,1),(2,3),(4,5),(6,7)
      Each gives a 2x2x4 problem (2 A-rows x 2 B-rows x 4 B-cols) -> rank 14 each
      Total: 4 * 14 = 56
    
    - Bottom row of A (i=2), same 4 column-pairs
      Each gives a 1x2x4 problem -> rank 8 each
      Total: 4 * 8 = 32
    
    Total rank: 56 + 32 = 88
    """
    total_rank = 88
    U = [[0]*total_rank for _ in range(12)]  # 3*4
    V = [[0]*total_rank for _ in range(32)]  # 4*8
    W = [[0]*total_rank for _ in range(24)]  # 8*3
    
    offset = 0
    
    # 4 blocks of 2x2x4 for top 2 rows of A
    # A rows: i=0,1; A cols: j=0,1,2,3 -> U rows 0..3 (i*m+j, m=4)
    #   Actually U row index = i*m + j = i*4 + j
    #   i=0: rows 0,1,2,3
    #   i=1: rows 4,5,6,7
    #   i=2: rows 8,9,10,11
    
    # B: rows k=0,1,2,3; cols l=0..7 -> V row index = k*p + l = k*8 + l
    # C: rows i=0,1,2; cols l=0..7 -> W row index = l*n + i = l*3 + i
    
    # For each of 4 column-pairs of B (pairs of output columns l):
    #   Pair 0: l=0,1 -> B cols 0,1
    #   Pair 1: l=2,3 -> B cols 2,3
    #   Pair 2: l=4,5 -> B cols 4,5
    #   Pair 3: l=6,7 -> B cols 6,7
    
    # For 2x2x4 blocks, we also need to split B rows into pairs:
    #   B rows k=0,1 and k=2,3
    #   So each 2x2x4 block uses 2 B-rows and 4 B-cols (but actually 2 B-cols per 2x2x2)
    
    # Let me reconsider. 2x2x4 means:
    #   2 A-rows (i=0,1), 2 B-rows (a pair of k), 4 B-cols (a pair of l)
    #   Wait, that's 2x2x4? No.
    #   A is 2x? B is ?x4. 
    #   Actually: A is (2 x 2) where the 2 cols of A correspond to 2 rows of B.
    #   B is (2 x 4) where 2 rows and 4 cols.
    #   C is (2 x 4).
    #   Rank 14 = two Strassen 2x2x2 blocks, each handling 2 B-cols.
    
    # So for each of 4 (A-row-pair, B-row-pair, B-col-pair) combinations?
    # No. Let me think more carefully.
    
    # A is 3x4. B is 4x8. C is 3x8.
    # C[i,l] = sum_k A[i,k] * B[k,l]
    
    # Top 2 rows of A: i=0,1. For each i in {0,1} and each l in {0..7}:
    #   C[i,l] = A[i,0]*B[0,l] + A[i,1]*B[1,l] + A[i,2]*B[2,l] + A[i,3]*B[3,l]
    
    # We can group by B-row-pairs: (k=0,1) and (k=2,3)
    # And by B-col-pairs: (l=0,1), (l=2,3), (l=4,5), (l=6,7)
    
    # For fixed B-row-pair (say k=0,1) and B-col-pair (say l=0,1):
    #   C[0,0] += A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #   C[0,1] += A[0,0]*B[0,1] + A[0,1]*B[1,1]
    #   C[1,0] += A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #   C[1,1] += A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # This is a 2x2x2 problem! (A is 2x2, B is 2x2, C is 2x2)
    # Rank 7.
    
    # So for each of 2 B-row-pairs and 4 B-col-pairs: 8 instances of 2x2x2.
    # Each rank 7. Total: 8*7 = 56.
    
    # Bottom row of A: i=2.
    #   C[2,l] = A[2,0]*B[0,l] + A[2,1]*B[1,l] + A[2,2]*B[2,l] + A[2,3]*B[3,l]
    # Group by B-row-pairs and B-col-pairs:
    # For fixed B-row-pair (k=0,1) and B-col-pair (l=0,1):
    #   C[2,0] += A[2,0]*B[0,0] + A[2,1]*B[1,0]
    #   C[2,1] += A[2,0]*B[0,1] + A[2,1]*B[1,1]
    # This is 1x2x2 (A is 1x2, B is 2x2, C is 1x2). Rank 4.
    # 2 B-row-pairs * 4 B-col-pairs = 8 instances, each rank 4. Total: 32.
    
    # Grand total: 56 + 32 = 88. 
    
    # Now let's construct.
    
    # For 2x2x2 blocks (Strassen):
    # A indices: (i,k) where i in {0,1}, k in {0,1} (relative)
    # B indices: (k,l) where k in {0,1}, l in {0,1} (relative)
    # C indices: (i,l) where i in {0,1}, l in {0,1} (relative)
    
    U_str, V_str, W_str = _strassen_2x2x2()
    
    # 1x2x2 schoolbook:
    # A: 1x2, B: 2x2, C: 1x2, rank 4
    U_122 = [[0]*4 for _ in range(2)]  # a0, a1
    V_122 = [[0]*4 for _ in range(4)]  # b00, b01, b10, b11
    W_122 = [[0]*4 for _ in range(2)]  # c0, c1
    for k in range(2):
        for l in range(2):
            col = k*2 + l
            U_122[k][col] = 1
            V_122[k*2+l][col] = 1
            W_122[l][col] = 1
    
    offset = 0
    for kp in range(2):  # B row pair: (0,1) or (2,3)
        k0 = kp * 2
        k1 = kp * 2 + 1
        for lp in range(4):  # B col pair: (0,1),(2,3),(4,5),(6,7)
            l0 = lp * 2
            l1 = lp * 2 + 1
            
            # 2x2x2 Strassen block for A rows i=0,1
            for r in range(4):
                i_rel = r // 2
                k_rel = r % 2
                i = i_rel  # 0 or 1
                k = k0 + k_rel
                U[i*4 + k][offset] = U_str[r][0] if False else 0
                # Actually U_str has 7 columns. Let me redo.
            
            # Let me just copy the 7 rank-1 terms directly.
            for c in range(7):
                # U rows: a00, a01, a10, a11 -> i=0,k=0; i=0,k=1; i=1,k=0; i=1,k=1
                for r in range(4):
                    i_rel = r // 2
                    k_rel = r % 2
                    i = i_rel
                    k = k0 + k_rel
                    U[i*4 + k][offset + c] += U_str[r][c]
                
                # V rows: b00, b01, b10, b11 -> k=0,l=0; k=0,l=1; k=1,l=0; k=1,l=1
                for r in range(4):
                    k_rel = r // 2
                    l_rel = r % 2
                    k = k0 + k_rel
                    l = l0 + l_rel
                    V[k*8 + l][offset + c] += V_str[r][c]
                
                # W rows: c00, c01, c10, c11 -> i=0,l=0; i=0,l=1; i=1,l=0; i=1,l=1
                for r in range(4):
                    i_rel = r // 2
                    l_rel = r % 2
                    i = i_rel
                    l = l0 + l_rel
                    W[l*3 + i][offset + c] += W_str[r][c]
            
            offset += 7
            
            # 1x2x2 schoolbook block for A row i=2
            for c in range(4):
                for r in range(2):
                    k_rel = r
                    k = k0 + k_rel
                    U[2*4 + k][offset + c] += U_122[r][c]
                
                for r in range(4):
                    k_rel = r // 2
                    l_rel = r % 2
                    k = k0 + k_rel
                    l = l0 + l_rel
                    V[k*8 + l][offset + c] += V_122[r][c]
                
                for r in range(2):
                    l_rel = r
                    l = l0 + l_rel
                    W[l*3 + 2][offset + c] += W_122[r][c]
            
            offset += 4
    
    # offset should be 8*7 + 8*4 = 56 + 32 = 88
    assert offset == total_rank, f"Expected {total_rank}, got {offset}"
    
    return {"U": U, "V": V, "W": W}
