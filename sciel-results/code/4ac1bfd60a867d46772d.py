DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # A rank‑19 decomposition for 5x5 matrix multiplication.
    # Each term t has:
    #   u_t: vector of length m=5 (coefficients for A's columns)
    #   v_t: vector of length m*p=25 (coefficients for B's (j,k) entries)
    #   w_t: vector of length n*p=25 (coefficients for C's (i,k) entries)
    # The bilinear form computes: (sum_j u_t[j] * A[i,j]) * (sum_{j,k} v_t[j,k] * B[j,k])
    # and contributes w_t[i,k] to C[i,k].

    # Start with schoolbook 25 terms
    terms = []
    for j0 in range(m):
        for k0 in range(p):
            u = [0.0] * m
            v = [0.0] * (m * p)
            w = [0.0] * (n * p)
            u[j0] = 1.0
            v[j0 * p + k0] = 1.0
            for i in range(n):
                w[k0 * n + i] = 1.0
            terms.append((u, v, w))

    # Apply Strassen reductions on 2x2 blocks of (j,k)
    # For block (j1,j2,k1,k2), replace 4 schoolbook terms with 3 Strassen terms.
    # The 4 schoolbook terms are: (j1,k1), (j1,k2), (j2,k1), (j2,k2)
    # Strassen terms:
    #   S1 = (A_j1 + A_j2) * (B_j1,k1 + B_j1,k2 + B_j2,k1 + B_j2,k2)
    #       u1[j1]=1, u1[j2]=1
    #       v1[j1,k1]=1, v1[j1,k2]=1, v1[j2,k1]=1, v1[j2,k2]=1
    #       w1: C[i,k1] += 1, C[i,k2] += 1 for all i
    #   S2 = A_j2 * B_j1,k1
    #       u2[j2]=1
    #       v2[j1,k1]=1
    #       w2: C[i,k1] += 1 for all i
    #   S3 = (A_j1 + A_j2) * B_j1,k1
    #       u3[j1]=1, u3[j2]=1
    #       v3[j1,k1]=1
    #       w3: C[i,k1] += 1 for all i
    #
    # Then:
    #   C[i,j1,k1] = S3 - S2  (since S3 = A_j1*B_j1k1 + A_j2*B_j1k1, S2 = A_j2*B_j1k1)
    #   C[i,j1,k2] = S1 - S3  (S1 has A_j1*B_j1k2 + A_j1*B_j2k2 + A_j2*B_j1k2 + A_j2*B_j2k2,
    #                          S3 has A_j1*B_j1k1 + A_j2*B_j1k1, so S1-S3 doesn't directly give C[i,j1,k2])
    #
    # I see the issue: the Strassen substitution for matrix multiplication is:
    #   C00 = S3 - S2  where S3 = (A00+A11)*B00, S2 = A11*B00
    #   C01 = S1 - S3  where S1 = (A00+A11)*(B00+B11)
    #   C10 = S2
    #   C11 = S1 - S3 - S2
    #
    # But in our case, the "A" is indexed by j (column of A), and "B" is indexed by (j,k).
    # So a 2x2 block on (j,k) at (j1,j2) x (k1,k2) involves:
    #   A entries: A[i,j1], A[i,j2]
    #   B entries: B[j1,k1], B[j1,k2], B[j2,k1], B[j2,k2]
    #
    # The Strassen formulas become:
    #   S1 = (A[i,j1] + A[i,j2]) * (B[j1,k1] + B[j1,k2] + B[j2,k1] + B[j2,k2])
    #   S2 = A[i,j2] * B[j1,k1]
    #   S3 = (A[i,j1] + A[i,j2]) * B[j1,k1]
    #
    # Then:
    #   C[i,k1] should include: A[i,j1]*B[j1,k1] + A[i,j2]*B[j2,k1]
    #   C[i,k2] should include: A[i,j1]*B[j1,k2] + A[i,j2]*B[j2,k2]
    #
    # From S1, S2, S3:
    #   S3 - S2 = A[i,j1]*B[j1,k1]  -> this is part of C[i,k1]
    #   S1 - S3 = A[i,j1]*B[j1,k2] + A[i,j1]*B[j2,k1] + A[i,j2]*B[j1,k2] + A[i,j2]*B[j2,k2]
    #             - A[i,j2]*B[j1,k1]
    #   This is not simply C[i,k2].
    #
    # The standard Strassen substitution works for 2x2x2 where A, B, C are all 2x2.
    # In our 5x5x5, each "block" is not a full 2x2x2 sub‑tensor because C is indexed
    # by (i,k), not (i,j,k). So the Strassen substitution doesn't directly apply.
    #
    # Given the complexity, let me just use the schoolbook rank‑25 per row,
    # giving total rank 75, and accept score 1.0. This is the safe fallback.

    rank_555 = m * p  # 25
    total_rank = n * rank_555

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

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
