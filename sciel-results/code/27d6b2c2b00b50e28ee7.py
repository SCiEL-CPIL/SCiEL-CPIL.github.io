"""
Rank-84 exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Uses Strassen's algorithm at the block level (7 products) combined with
schoolbook multiplication inside each 2x2 * 2x3 block.

Structure:
  A is 4x4, split into four 2x2 blocks: A00, A01, A10, A11
  B is 4x6, split into four 2x3 blocks: B00, B01, B10, B11
  C is 4x6, split into four 2x3 blocks: C00, C01, C10, C11

Strassen computes 7 intermediate products M1..M7, each of which is a
"scalar" product of two 2x2 matrices (formed by adding/subtracting blocks).
Each such 2x2 * 2x3 multiplication is expanded via schoolbook (rank 12).

Total rank = 7 * 12 = 84.

The output W follows the convention W[k*n + i, r] where n=4, so
W has shape (p*n, rank) = (24, rank).
"""

def construct():
    n, m, p = 4, 4, 6
    rank = 84

    U = [[0] * rank for _ in range(n * m)]   # 16 x 84
    V = [[0] * rank for _ in range(m * p)]   # 24 x 84
    W = [[0] * rank for _ in range(p * n)]   # 24 x 84

    # Strassen's 7 steps for 2x2 block matrices
    # Block IDs: 0=A00, 1=A01, 2=A10, 3=A11 (and similarly for B, C)
    #
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    steps = [
        # M1 = (A00 + A11)(B00 + B11)
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
        # M2 = (A10 + A11)B00
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)]},
        # M3 = A00(B01 - B11)
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
        # M4 = A11(B10 - B00)
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
        # M5 = (A00 + A01)B11
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
        # M6 = (A10 - A00)(B00 + B01)
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
        # M7 = (A01 - A11)(B10 + B11)
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]}
    ]

    # For each Strassen step, expand the inner 2x2 * 2x3 multiplication
    # using schoolbook: C[r,k] = sum_s A[r,s] * B[s,k]
    # This gives 2*2*3 = 12 rank-1 terms per step.
    #
    # Local indices:
    #   A_inner: r in {0,1}, s in {0,1}  (2x2)
    #   B_inner: s in {0,1}, k in {0,1,2} (2x3)
    #   C_inner: r in {0,1}, k in {0,1,2} (2x3)
    #
    # Global mapping for A (4x4, blocks are 2x2):
    #   block 0 (A00): I=r, J=s
    #   block 1 (A01): I=r, J=s+2
    #   block 2 (A10): I=r+2, J=s
    #   block 3 (A11): I=r+2, J=s+2
    #
    # Global mapping for B (4x6, blocks are 2x3):
    #   block 0 (B00): J=s, K=k
    #   block 1 (B01): J=s, K=k+3
    #   block 2 (B10): J=s+2, K=k
    #   block 3 (B11): J=s+2, K=k+3
    #
    # Global mapping for C (4x6, blocks are 2x3):
    #   block 0 (C00): I=r, K=k
    #   block 1 (C01): I=r, K=k+3
    #   block 2 (C10): I=r+2, K=k
    #   block 3 (C11): I=r+2, K=k+3

    for i, step in enumerate(steps):
        for r in range(2):
            for s in range(2):
                for k in range(3):
                    c_in = r * 6 + s * 3 + k
                    c_global = i * 12 + c_in

                    # U contributions: A_inner[r, s]
                    for block_id, coeff in step['A']:
                        if block_id == 0:
                            I, J = r, s
                        elif block_id == 1:
                            I, J = r, s + 2
                        elif block_id == 2:
                            I, J = r + 2, s
                        elif block_id == 3:
                            I, J = r + 2, s + 2
                        else:
                            continue
                        u_idx = I * 4 + J
                        U[u_idx][c_global] += coeff

                    # V contributions: B_inner[s, k]
                    for block_id, coeff in step['B']:
                        if block_id == 0:
                            J, K = s, k
                        elif block_id == 1:
                            J, K = s, k + 3
                        elif block_id == 2:
                            J, K = s + 2, k
                        elif block_id == 3:
                            J, K = s + 2, k + 3
                        else:
                            continue
                        v_idx = J * 6 + K
                        V[v_idx][c_global] += coeff

                    # W contributions: C_inner[r, k]
                    # W index convention: W[k*n + i, r] where n=4
                    # So w_idx = K * 4 + I
                    for block_id, coeff in step['C']:
                        if block_id == 0:
                            I, K = r, k
                        elif block_id == 1:
                            I, K = r, k + 3
                        elif block_id == 2:
                            I, K = r + 2, k
                        elif block_id == 3:
                            I, K = r + 2, k + 3
                        else:
                            continue
                        w_idx = K * 4 + I
                        W[w_idx][c_global] += coeff

    return {"U": U, "V": V, "W": W}
