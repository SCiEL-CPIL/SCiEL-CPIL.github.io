"""Block-wise Strassen decomposition for 4x4 matrix multiplication.

Partitions 4x4 matrices into 2x2 blocks of 2x2 matrices.
Applies Strassen's 7-multiplication algorithm at the block level,
with each block multiplication done via the rank-4 schoolbook method.
Total rank: 7 * 4 = 28.
"""
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 4

    # Index mappings:
    # A is n x m = 4x4, indexed by (i, j) -> global index i*m + j = i*4 + j
    # B is m x p = 4x4, indexed by (j, k) -> global index j*p + k = j*4 + k
    # C (output W) is p x n = 4x4, indexed by (k, i) -> global index k*n + i = k*4 + i

    # We partition into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where each Ajk is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]

    # Block indices: A_ij has rows (2i, 2i+1) and cols (2j, 2j+1)
    # B_jk has rows (2j, 2j+1) and cols (2k, 2k+1)
    # C_ki has rows (2k, 2k+1) and cols (2i, 2i+1)

    # Strassen's 7 products:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)

    # Output blocks:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    rank = 28  # 7 Strassen products * 4 schoolbook terms each

    U = [[0] * rank for _ in range(n * m)]  # 16 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28

    # Helper: for a 2x2 block at position (block_row, block_col) in a 4x4 matrix,
    # get the 4 global indices (row, col) -> row*4 + col
    def block_indices(base_row, base_col, rows, cols):
        """Get global indices for a 2x2 block starting at (base_row, base_col)."""
        idxs = []
        for dr in range(rows):
            for dc in range(cols):
                idxs.append((base_row + dr) * 4 + (base_col + dc))
        return idxs

    # Define the 7 Strassen products.
    # For each product s (0-6), we have:
    #   A_factor: a 4x4 matrix M_A_s such that vec(S_A_s) = M_A_s * vec(A)
    #   B_factor: a 4x4 matrix M_B_s such that vec(S_B_s) = M_B_s * vec(B)
    #   C_factor: a 16x4 matrix M_C_s such that the contribution to C is M_C_s * vec(P_s)
    # where S_A_s and S_B_s are 2x2 matrices (vec'd to length 4).

    # Actually, let's think of it differently.
    # For each Strassen product s and each schoolbook term r (0-3):
    #   The schoolbook term r for 2x2*2x2 -> 2x2 is:
    #     (S_A)_r_row * (S_B)_r_col contributes to (P)_r_row, r_col
    #   In the global tensor:
    #     U[i*4+j, col] += M_A_s[i_local, r_row] * (1 for the A element at local (i_local, r_row))
    #     V[j*4+k, col] += M_B_s[j_local, r_col]
    #     W[k*4+i, col] += M_C_s[k_local*2 + i_local, r_row*2 + r_col] ... 

    # Let me be more explicit.
    # For a 2x2 * 2x2 -> 2x2 multiplication via schoolbook:
    # P[a,b] = sum_{c=0}^{1} S_A[a,c] * S_B[c,b]
    # The 4 schoolbook terms correspond to c=0 and c=1, but actually each (a,c) pair
    # gives one term for each b. So 4 terms: for each c in {0,1} and for the output,
    # each term is S_A[a,c] * S_B[c,b] added to P[a,b].
    # In rank-4 schoolbook: for each c in {0,1} and for each (a,b):
    #   Actually the standard rank-4 is: 4 multiplications, one for each c in {0,1} 
    #   but that's only 2... no, for 2x2*2x2, schoolbook is 8 multiplications.
    #   Rank-4 means we use 4 scalar multiplications. That's Strassen-like for 2x2?
    #   No, the minimum rank for 2x2 matrix multiplication is 7 (Strassen).
    #   Wait, the task says "rank-4 schoolbook algorithm" for 2x2 matrices.
    #   But 2x2 matrix multiplication has rank at least 7. You can't do it with 4.
    #   
    #   Hmm, let me re-read: "treating the multiplication as a matrix multiplication 
    #   over the ring of 2x2 matrices, where the seven Strassen scalar multiplications 
    #   are replaced by 2x2 matrix multiplications computed via the rank-4 schoolbook 
    #   algorithm, yielding a total rank of 28"
    #
    #   Wait, 2x2 matrix multiplication cannot be done with rank 4. The schoolbook 
    #   for 2x2*2x2 uses 8 multiplications. Unless "rank-4" refers to something else.
    #   
    #   Actually, I think the directive might be slightly confused, but let's just use
    #   the standard schoolbook for 2x2*2x2 which uses 8 multiplications. That would give
    #   7 * 8 = 56, which is worse than 64.
    #
    #   Alternatively, maybe they mean we use the 4-multiplication algorithm for 
    #   multiplying two 2x2 matrices when one of them is diagonal or something? No.
    #
    #   Let me reconsider. Perhaps "rank-4 schoolbook" means we decompose each 2x2*2x2
    #   into 4 terms by grouping. But that's not exact.
    #
    #   Actually, I think the intended approach is:
    #   Use Strassen at the top level (7 products of 2x2 matrices).
    #   For each 2x2*2x2, use the standard 8-multiplication schoolbook.
    #   Total rank = 7 * 8 = 56.
    #
    #   But the directive says rank 28. Let me just implement with 8 multiplications 
    #   per 2x2*2x2 to get rank 56, which is still better than 64.
    #
    #   Actually wait - maybe we can do better. Let me just implement the 56-rank version
    #   since 56 < 64.

    rank = 56  # 7 * 8

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # For each of the 7 Strassen products, we compute S_A * S_B where S_A, S_B are 2x2.
    # We use schoolbook: for each (a, c, b) in {0,1}^3, we have a multiplication
    # S_A[a,c] * S_B[c,b] added to P[a,b].
    # That's 8 terms per Strassen product.

    # Define Strassen factors.
    # S_A_s = M_A_s * vec(A) where M_A_s is 4x16
    # S_B_s = M_B_s * vec(B) where M_B_s is 4x16
    # C gets contribution M_C_s * vec(P_s) where M_C_s is 16x4

    # Block positions: A_ij is at rows 2i..2i+1, cols 2j..2j+1
    # Local index within block: (lr, lc) -> global (2i+lr, 2j+lc) -> index (2i+lr)*4 + (2j+lc)

    def a_idx(i, j):
        return i * 4 + j

    def b_idx(j, k):
        return j * 4 + k

    def c_idx(k, i):
        return k * 4 + i

    # Strassen definitions:
    # For product s, A_factor is a 4x16 matrix (vec(S_A_s) = M_A_s vec(A))
    # S_A_s is 2x2, vec'd as [S_A[0,0], S_A[0,1], S_A[1,0], S_A[1,1]]

    # P1: (A00+A11) * (B00+B11)
    # A00: indices (0,0),(0,1),(1,0),(1,1) -> a_idx(0,0),a_idx(0,1),a_idx(1,0),a_idx(1,1)
    # A11: indices (2,2),(2,3),(3,2),(3,3) -> a_idx(2,2),a_idx(2,3),a_idx(3,2),a_idx(3,3)
    # M_A_1: 4x16, rows correspond to S_A[0,0],S_A[0,1],S_A[1,0],S_A[1,1]
    # S_A[0,0] = A[0,0]+A[2,2]
    # S_A[0,1] = A[0,1]+A[2,3]
    # S_A[1,0] = A[1,0]+A[3,2]
    # S_A[1,1] = A[1,1]+A[3,3]

    def make_M_A(entries):
        """entries: list of 4 tuples (val, a_row, a_col) for each row of the 4x16 matrix"""
        M = [[0.0]*16 for _ in range(4)]
        for r, (val, ar, ac) in enumerate(entries):
            M[r][a_idx(ar, ac)] = val
        return M

    def make_M_B(entries):
        """entries: list of 4 tuples (val, b_row, b_col)"""
        M = [[0.0]*16 for _ in range(4)]
        for r, (val, br, bc) in enumerate(entries):
            M[r][b_idx(br, bc)] = val
        return M

    # For C factor: M_C_s is 16x4.
    # vec(P_s) = [P[0,0], P[0,1], P[1,0], P[1,1]]
    # The contribution to C: C_block = P_s placed in the appropriate block position.
    # For C_ki (block at rows 2k..2k+1, cols 2i..2i+1):
    # M_C_s[c_idx(2k+lr, 2i+lc), r_local] where r_local is 0..3 for P[lr,lc]

    def make_M_C(block_row, block_col):
        """M_C is 16x4, mapping vec(P) to global C positions for block (block_row, block_col)"""
        M = [[0.0]*4 for _ in range(16)]
        for lr in range(2):
            for lc in range(2):
                r_local = lr * 2 + lc
                M[c_idx(2*block_row + lr, 2*block_col + lc)][r_local] = 1.0
        return M

    # Define the 7 Strassen products
    # Each entry: (M_A, M_B, list_of_M_C)
    # Multiple M_C for same product? No, each product goes to specific C blocks.
    # Actually, in Strassen, each P_s contributes to multiple C blocks.
    # So M_C_s is actually a sum of contributions to different blocks.
    # Let me just build M_C_s as 16x4 directly.

    def make_M_C_combined(blocks):
        """blocks: list of (block_row, block_col, coefficient)"""
        M = [[0.0]*4 for _ in range(16)]
        for br, bc, coeff in blocks:
            for lr in range(2):
                for lc in range(2):
                    r_local = lr * 2 + lc
                    M[c_idx(2*br + lr, 2*bc + lc)][r_local] += coeff
        return M

    # Strassen products:
    # P1 = (A00+A11)*(B00+B11) -> C00 += P1, C11 += P1
    strassen = []

    # P1
    M_A = make_M_A([(1,0,0),(1,0,1),(1,1,0),(1,1,1),
                    (1,2,2),(1,2,3),(1,3,2),(1,3,3)])
    # Wait, make_M_A takes 4 entries, one per row of the 4x16 matrix.
    # Row 0: S_A[0,0] = A[0,0]+A[2,2]
    # Row 1: S_A[0,1] = A[0,1]+A[2,3]
    # Row 2: S_A[1,0] = A[1,0]+A[3,2]
    # Row 3: S_A[1,1] = A[1,1]+A[3,3]
    # But make_M_A only sets one entry per row. I need to handle sums.

    # Let me redo this more carefully.

    def make_M_A_general(coeffs):
        """coeffs: list of 4 lists, each list has (val, ar, ac) tuples"""
        M = [[0.0]*16 for _ in range(4)]
        for r in range(4):
            for val, ar, ac in coeffs[r]:
                M[r][a_idx(ar, ac)] += val
        return M

    def make_M_B_general(coeffs):
        M = [[0.0]*16 for _ in range(4)]
        for r in range(4):
            for val, br, bc in coeffs[r]:
                M[r][b_idx(br, bc)] += val
        return M

    # P1: (A00+A11)*(B00+B11)
    M_A_1 = make_M_A_general([
        [(1,0,0),(1,2,2)],
        [(1,0,1),(1,2,3)],
        [(1,1,0),(1,3,2)],
        [(1,1,1),(1,3,3)],
    ])
    M_B_1 = make_M_B_general([
        [(1,0,0),(1,2,2)],
        [(1,0,1),(1,2,3)],
        [(1,1,0),(1,3,2)],
        [(1,1,1),(1,3,3)],
    ])
    M_C_1 = make_M_C_combined([(0,0,1),(1,1,1)])
    strassen.append((M_A_1, M_B_1, M_C_1))

    # P2: (A10+A11)*B00
    M_A_2 = make_M_A_general([
        [(1,2,0),(1,2,2)],
        [(1,2,1),(1,2,3)],
        [(1,3,0),(1,3,2)],
        [(1,3,1),(1,3,3)],
    ])
    M_B_2 = make_M_B_general([
        [(1,0,0)],
        [(1,0,1)],
        [(1,1,0)],
        [(1,1,1)],
    ])
    M_C_2 = make_M_C_combined([(1,0,1),(1,1,-1)])
    strassen.append((M_A_2, M_B_2, M_C_2))

    # P3: A00*(B01-B11)
    M_A_3 = make_M_A_general([
        [(1,0,0)],
        [(1,0,1)],
        [(1,1,0)],
        [(1,1,1)],
    ])
    M_B_3 = make_M_B_general([
        [(1,0,1),(-1,2,1)],
        [(1,0,2),(-1,2,2)],
        [(1,1,1),(-1,3,1)],
        [(1,1,2),(-1,3,2)],
    ])
    M_C_3 = make_M_C_combined([(0,1,1),(1,1,1)])
    strassen.append((M_A_3, M_B_3, M_C_3))

    # P4: A11*(B10-B00)
    M_A_4 = make_M_A_general([
        [(1,2,2)],
        [(1,2,3)],
        [(1,3,2)],
        [(1,3,3)],
    ])
    M_B_4 = make_M_B_general([
        [(1,2,0),(-1,0,0)],
        [(1,2,1),(-1,0,1)],
        [(1,3,0),(-1,1,0)],
        [(1,3,1),(-1,1,1)],
    ])
    M_C_4 = make_M_C_combined([(0,0,1),(1,0,1)])
    strassen.append((M_A_4, M_B_4, M_C_4))

    # P5: (A00+A01)*B11
    M_A_5 = make_M_A_general([
        [(1,0,0),(1,0,2)],
        [(1,0,1),(1,0,3)],
        [(1,1,0),(1,1,2)],
        [(1,1,1),(1,1,3)],
    ])
    M_B_5 = make_M_B_general([
        [(1,2,2)],
        [(1,2,3)],
        [(1,3,2)],
        [(1,3,3)],
    ])
    M_C_5 = make_M_C_combined([(0,0,-1),(0,1,1)])
    strassen.append((M_A_5, M_B_5, M_C_5))

    # P6: (A10-A00)*(B00+B01)
    M_A_6 = make_M_A_general([
        [(1,2,0),(-1,0,0)],
        [(1,2,1),(-1,0,1)],
        [(1,3,0),(-1,1,0)],
        [(1,3,1),(-1,1,1)],
    ])
    M_B_6 = make_M_B_general([
        [(1,0,0),(1,0,2)],
        [(1,0,1),(1,0,3)],
        [(1,1,0),(1,1,2)],
        [(1,1,1),(1,1,3)],
    ])
    M_C_6 = make_M_C_combined([(1,1,1)])
    strassen.append((M_A_6, M_B_6, M_C_6))

    # P7: (A01-A11)*(B10+B11)
    M_A_7 = make_M_A_general([
        [(1,0,2),(-1,2,2)],
        [(1,0,3),(-1,2,3)],
        [(1,1,2),(-1,3,2)],
        [(1,1,3),(-1,3,3)],
    ])
    M_B_7 = make_M_B_general([
        [(1,2,0),(1,2,2)],
        [(1,2,1),(1,2,3)],
        [(1,3,0),(1,3,2)],
        [(1,3,1),(1,3,3)],
    ])
    M_C_7 = make_M_C_combined([(0,0,1)])
    strassen.append((M_A_7, M_B_7, M_C_7))

    # Now fill in U, V, W
    col = 0
    for s in range(7):
        M_A_s, M_B_s, M_C_s = strassen[s]
        # For each schoolbook term: for c in {0,1}, for a in {0,1}, for b in {0,1}
        # P[a,b] += S_A[a,c] * S_B[c,b]
        # So for each (a, c, b): one multiplication term
        for c in range(2):
            for a in range(2):
                for b in range(2):
                    # U contribution: S_A[a,c] = sum over A entries
                    # V contribution: S_B[c,b] = sum over B entries
                    # W contribution: P[a,b] goes to C block position
                    # P[a,b] is at local position (a,b) in the 2x2 result
                    # Which C block? Depends on which Strassen product, but
                    # M_C_s already encodes where P_s goes.
                    # Actually M_C_s maps vec(P_s) to global C.
                    # vec(P_s) = [P[0,0], P[0,1], P[1,0], P[1,1]]
                    # P[a,b] is at index a*2+b in vec(P_s)
                    p_idx = a * 2 + b

                    # U: for each A element that contributes to S_A[a,c]
                    for ar in range(2):
                        for ac in range(2):
                            val = M_A_s[a*2+c if False else (a, c)]  # need to index properly
                    # Let me just iterate over all 16 A positions
                    for ai in range(4):
                        for aj in range(4):
                            ua = M_A_s[a*2 + c][ai*4 + aj]  # This is wrong, M_A_s is 4x16
                            # M_A_s has 4 rows (for S_A[0,0],S_A[0,1],S_A[1,0],S_A[1,1])
                            # and 16 cols (for A[0,0]..A[3,3])
                            # S_A[a,c] is row (a*2+c) of M_A_s
                            # A[ai,aj] is col (ai*4+aj) of M_A_s
                            # This only works if M_A_s is indexed as [a*2+c, ai*4+aj]
                            pass

    # Let me redo the filling more carefully.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for s in range(7):
        M_A_s, M_B_s, M_C_s = strassen[s]
        for c in range(2):
            for a in range(2):
                for b in range(2):
                    p_idx = a * 2 + b
                    # S_A[a,c] is row (a*2+c) of M_A_s (4x16)
                    sa_row = a * 2 + c
                    # S_B[c,b] is row (c*2+b) of M_B_s (4x16)
                    sb_row = c * 2 + b

                    # U: for each A position (ai,aj)
                    for ai in range(4):
                        for aj in range(4):
                            val = M_A_s[sa_row][ai * 4 + aj]
                            if val != 0:
                                U[ai * 4 + aj][col] += val

                    # V: for each B position (bi,bj)
                    for bi in range(4):
                        for bj in range(4):
                            val = M_B_s[sb_row][bi * 4 + bj]
                            if val != 0:
                                V[bi * 4 + bj][col] += val

                    # W: for each C position (ci,cj)
                    for ci in range(4):
                        for cj in range(4):
                            val = M_C_s[ci * 4 + cj][p_idx]
                            if val != 0:
                                W[ci * 4 + cj][col] += val

                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
