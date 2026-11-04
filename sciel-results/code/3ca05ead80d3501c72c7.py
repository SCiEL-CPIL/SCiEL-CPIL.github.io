"""Rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = 22
    U = [[0] * rank for _ in range(n * m)]   # 12 x 22
    V = [[0] * rank for _ in range(m * p)]   # 32 x 22
    W = [[0] * rank for _ in range(p * n)]   # 24 x 22

    # ---- Block A: output row i=0, 1x4 @ 4x8 as 8 rank-1 terms (cols 0..7) ----
    # For each output column k: U[0*m+j][k]=1 (j=0..3), V[j*p+k][k]=1, W[k*n+0][k]=1
    for k in range(p):
        for j in range(m):
            U[0 * m + j][k] = 1
            V[j * p + k][k] = 1
        W[k * n + 0][k] = 1

    # ---- Block B: output rows i=1,2, 2x4 @ 4x8 via Strassen (cols 8..21) ----
    # Inner dim 4 split: a = cols {0,1}, b = cols {2,3} (each 2-dim)
    # 7 Strassen products for the 2x2 block; each product is a 2x2 (inner) x 8 (outer)
    # matrix, expanded as 16 rank-1 terms.
    a_cols = [0, 1]
    b_cols = [2, 3]
    out_rows = [1, 2]

    # (i_index, j_index, s, t, sign) for the 7 Strassen products
    # P1=(a+b)(c+d), P2=(a-b)(c+d), P3=a(e-f), P4=b(d-e), P5=(a+c)(f-d),
    # P6=(b+d)(e-c), P7=(a-d)(c+e)
    strassen = [
        (0, 0, 0, 0, 1),   # P1: a*c
        (0, 0, 0, 1, 1),   # P1: a*d
        (1, 0, 0, 0, 1),   # P1: b*c
        (1, 0, 0, 1, 1),   # P1: b*d
        (0, 0, 1, 0, 1),   # P2: a*c
        (0, 0, 1, 1, 1),   # P2: a*d
        (1, 0, 1, 0, -1),  # P2: b*c
        (1, 0, 1, 1, -1),  # P2: b*d
        (0, 1, 0, 0, 1),   # P3: a*e
        (0, 1, 0, 1, -1),  # P3: a*f
        (1, 1, 0, 0, 1),   # P4: b*d
        (1, 1, 0, 1, -1),  # P4: b*e
        (0, 2, 0, 0, 1),   # P5: a*f
        (0, 2, 0, 1, -1),  # P5: a*d
        (0, 2, 1, 0, 1),   # P5: c*f
        (0, 2, 1, 1, -1),  # P5: c*d
        (1, 3, 0, 0, 1),   # P6: b*e
        (1, 3, 0, 1, 1),   # P6: b*f
        (1, 3, 1, 0, -1),  # P6: d*e
        (1, 3, 1, 1, -1),  # P6: d*f
        (0, 4, 0, 0, 1),   # P7: a*c
        (0, 4, 0, 1, 1),   # P7: a*e
        (0, 4, 1, 0, -1),  # P7: d*c
        (0, 4, 1, 1, -1),  # P7: d*e
    ]

    # Strassen combination for the 2x2 output:
    # X = P1 + P5 - P6 + P7  (rows 1,1)
    # Y = P3 + P7            (rows 1,2)
    # Z = P4 + P6            (rows 2,1)
    # T = P2 + P1            (rows 2,2)
    # (i_out, j_out, s, t, sign)
    combine = [
        (0, 0, 0, 0, 1),   # X += P1
        (0, 0, 2, 0, 1),   # X += P5
        (0, 0, 3, 0, -1),  # X -= P6
        (0, 0, 4, 0, 1),   # X += P7
        (1, 0, 2, 0, 1),   # Y += P5
        (1, 0, 4, 0, 1),   # Y += P7
        (0, 1, 1, 0, 1),   # Z += P3
        (0, 1, 3, 0, 1),   # Z += P6
        (1, 1, 1, 0, 1),   # T += P2
        (1, 1, 0, 0, 1),   # T += P1
    ]

    # Build the 16-term expansion for each (i_out, j_out, s, t) and accumulate
    # contributions to columns 8..21 via the combine table.
    # We need a mapping from (s, t, i_out, j_out, k) to a column in 8..21.
    # There are 7*2*2*2*8 = 448 (s,t,i_out,j_out,k) combos but each (s,t) shares
    # its 16 terms. Let's index columns by (s, t, i_out, j_out, k).
    # Actually each (s,t) product has 16 terms (i_out in 0,1; j_out in 0,1; k in 0..7).
    # That's 7 * 16 = 112 > 14. So Strassen as described gives 112 terms, not 14!
    # The "rank-14" claim is wrong for this expansion. Let me reconsider.

    # The 2x4 @ 4x8 product has 16 outputs. Strassen reduces the 7 "block products"
    # but each block product is 2x8 = 16 scalar products = 16 rank-1 terms.
    # So 7 * 16 = 112 rank-1 terms, total rank = 8 + 112 = 120. That's worse.

    # Let me instead just do the direct expansion for block B: 2x4@4x8 = 16 rank-1.
    # Total = 8 + 16 = 24. Still better than 96.
    # Actually let me reconsider: can we share? The 1-row block uses 8 terms.
    # The 2-row block: 2*4*8 = 64 scalar products but as rank-1 in the tensor sense,
    # each (i,j,k) triple is one rank-1 term. 2*4*8 = 64 terms. Total = 8 + 64 = 72.

    # Let me just do the full direct (schoolbook) for block B and outer for block A.
    # Reset and do it cleanly.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Block A: i=0, rank 8 (cols 0..7)
    for k in range(p):
        for j in range(m):
            U[0 * m + j][k] = 1
            V[j * p + k][k] = 1
        W[k * n + 0][k] = 1

    # Block B: i=1,2, direct rank 2*4*8=64 (cols 8..71) -- but rank is only 22!
    # I need rank >= 8 + 64 = 72. Let me set rank properly.

    # Actually the minimum possible: the tensor for 3x4@4x8 has rank at least
    # max(3*4, 4*8, 8*3) = 32 by the trivial lower bound (each output needs its
    # own... no). The true rank is unknown but <= 96 (schoolbook).
    # Let me just use schoolbook rank 96 to be safe and valid.

    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
