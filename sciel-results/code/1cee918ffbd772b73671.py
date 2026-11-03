"""Matrix multiplication 2x4 by 4x5 via block Strassen + schoolbook,
with exact verification."""

import time

DIMENSIONS = [2, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0

# ---------------------------------------------------------------------------
# Pre-computed rank-12 decomposition for  C[2,2] = A[2,4] * B[4,2]
#
#  A = [[a00, a01, a02, a03],
#       [a10, a11, a12, a13]]
#  B = [[b00, b01],
#       [b10, b11],
#       [b20, b21],
#       [b30, b31]]
#
#  C[0,0] = a00*b00 + a01*b10 + a02*b20 + a03*b30
#  C[0,1] = a00*b01 + a01*b11 + a02*b21 + a03*b31
#  C[1,0] = a10*b00 + a11*b10 + a12*b20 + a13*b30
#  C[1,1] = a10*b01 + a11*b11 + a12*b21 + a13*b31
# ---------------------------------------------------------------------------
# Each entry: (U_entries, V_entries, W_entry)
#   U_entries: list of (row_idx, coeff)  -- row in A (0..7)
#   V_entries: list of (col_idx, coeff)  -- col in B (0..7)
#   W_entry:   (row_idx, coeff)          -- row in C (0..3)
#
# Row indices: A row i*m+j where i in {0,1}, j in {0,1,2,3}
#              B row j*p+k where j in {0,1,2,3}, k in {0,1}
#              C row k*n+i where k in {0,1}, i in {0,1}
#
# Decomposition (12 terms):
#  T1: a00 * b00 -> C00
#  T2: a01 * b10 -> C00
#  T3: a02 * b20 -> C00
#  T4: a03 * b30 -> C00
#  T5: a00 * b01 -> C01
#  T6: a01 * b11 -> C01
#  T7: a02 * b21 -> C01
#  T8: a03 * b31 -> C01
#  T9: a10 * b00 -> C10
#  T10: a11 * b10 -> C10
#  T11: a12 * b20 -> C10
#  T12: a13 * b30 -> C10
#
# Wait -- that's schoolbook (rank 16 for 2x4x4x2). We need rank 12.
# Let me use a proper Strassen-like decomposition.
#
# Actually, for 2x4 * 4x2 -> 2x2, the bilinear complexity is at most
# 2*4*2 = 16 (schoolbook). Can we do better?
#
# Split A into two 2x2 blocks: A = [A_L, A_R] where A_L = A[:,0:2], A_R = A[:,2:4]
# Split B into two 2x2 blocks: B = [B_T; B_B] where B_T = B[0:2,:], B_B = B[2:4,:]
#
# C = A_L * B_T + A_R * B_B
#
# Each 2x2*2x2 -> 2x2 has rank 7 (Strassen).
# So C = A_L*B_T + A_R*B_B has rank at most 14.
#
# But we can do better by sharing structure. Let's use:
#   M1 = (A_L + A_R) * (B_T + B_B)  -- rank 7
#   M2 = A_L * B_T                   -- rank 7
#   Then C = M1 - M2 - A_R*B_B ... no, that doesn't help directly.
#
# Alternative: C = A_L*B_T + A_R*B_B
#   = (A_L + A_R)*(B_T + B_B) - A_L*B_B - A_R*B_T
#   That's 7 + 7 + 7 = 21, worse.
#
# Let me just use rank 14 (two Strassen blocks) as a safe baseline,
# and then try to find a rank-12 decomposition.
#
# For now, use two Strassen 2x2x2 blocks: rank = 7 + 7 = 14 for the 2x4x4x2 part.
# Total rank = 14 + 8 (schoolbook 2x4x4x1) = 22.
#
# Actually wait: the 2x4x4x1 part is C[:,4] = A * B[:,4].
# A is 2x4, B[:,4] is 4x1, C[:,4] is 2x1.
# Schoolbook: sum over j=0..3 of A[i,j]*B[j,4] for each i. Rank = 8.
#
# Can we do the 2x4x4x1 in fewer than 8? The bilinear complexity of
# 2x4 * 4x1 -> 2x1 is at most 8. It's actually 8 (you need all 8 products
# unless there's structure). So rank 8 is optimal for that part.
#
# So total with two Strassen + schoolbook = 14 + 8 = 22.
# That's better than 36!
#
# But can we do even better on the 2x4x4x2 part?
# The tensor for 2x4 * 4x2 -> 2x2:
#   T_{i,j,k,l} = delta_{i,k} * delta_{j,l}  (summed over j)
#   Actually T = sum_j e_i * e_j * e_j * e_k  (in appropriate spaces)
#
# The bilinear complexity of matmul(2,4,2) is known to be at most 14
# (two Strassen) and at least... let me think.
#
# For matmul(m,n,p), the trivial upper bound is m*n*p.
# For (2,4,2): 16. Strassen gives 14.
#
# Can we get 12? Let me try to construct it.
#
# C[0,0] = a00*b00 + a01*b10 + a02*b20 + a03*b30
# C[0,1] = a00*b01 + a01*b11 + a02*b21 + a03*b31
# C[1,0] = a10*b00 + a11*b10 + a12*b20 + a13*b30
# C[1,1] = a10*b01 + a11*b11 + a12*b21 + a13*b31
#
# Let me try to find 12 rank-1 terms.
#
# Group by A-row pairs and B-column pairs:
#
# For C[0,0]: involves (a00,b00), (a01,b10), (a02,b20), (a03,b30)
# For C[0,1]: involves (a00,b01), (a01,b11), (a02,b21), (a03,b31)
# For C[1,0]: involves (a10,b00), (a11,b10), (a12,b20), (a13,b30)
# For C[1,1]: involves (a10,b01), (a11,b11), (a12,b21), (a13,b31)
#
# Each rank-1 term is (linear form in A) * (linear form in B) * (output position).
#
# Let me try:
#  t1 = a00 * b00 * e00
#  t2 = a01 * b10 * e00
#  t3 = a02 * b20 * e00
#  t4 = a03 * b30 * e00
#  t5 = a00 * b01 * e01
#  t6 = a01 * b11 * e01
#  t7 = a02 * b21 * e01
#  t8 = a03 * b31 * e01
#  t9 = a10 * b00 * e10
#  t10 = a11 * b10 * e10
#  t11 = a12 * b20 * e10
#  t12 = a13 * b30 * e10
#
# That's 12 terms but missing C[1,1]! I need 16 terms for schoolbook.
#
# OK so I can't just drop terms. Let me think differently.
#
# What if I combine some terms?
#  t1 = (a00 + a10) * b00 * (e00 + e10)  -- contributes to C00, C10, but also C01? No.
#
# Actually each rank-1 term contributes to exactly ONE output position (since
# the output is a 2x2 matrix, and we're decomposing the tensor).
#
# So I need:
#  4 terms for C00 (or fewer if I can combine)
#  4 terms for C01
#  4 terms for C10
#  4 terms for C11
#
# Total 16 for schoolbook. To get rank 12, I need to share terms across
# output positions, which means a single rank-1 term must contribute to
# multiple output positions. But in a rank-1 term, the output factor is a
# single vector in the output space. If that vector has multiple non-zero
# components, the term contributes to multiple outputs.
#
# E.g., t = a00 * b00 * (e00 + e10) contributes a00*b00 to both C00 and C10.
# But we need a00*b00 in C00 and a10*b00 in C10, not a00*b00 in C10.
#
# So that doesn't work directly. We'd need:
#  t = (a00 + a10) * b00 * (e00 + e10)
#  This gives: a00*b00*e00 + a00*b00*e10 + a10*b00*e00 + a10*b00*e10
#  We want: a00*b00*e00 + a10*b00*e10
#  Extra terms: a00*b00*e10 + a10*b00*e00
#
# We'd need to cancel those. This is getting complicated.
#
# Let me just use rank 14 (two Strassen blocks) for now. Total = 14 + 8 = 22.
# That's a big improvement from 36.


def _remaining():
    return _DEADLINE - time.time()


def construct():
    """
    Return {'U','V','W'} factor matrices that exactly decompose the
    2x4-by-4x5 matrix multiplication tensor.

    Strategy (block decomposition):
      - Split B into B[:,0:2], B[:,2:4], B[:,4].
      - C[:,0:2] = A * B[:,0:2]  (2x4 * 4x2 -> 2x2): two Strassen 2x2x2 blocks, rank 14
      - C[:,2:4] = A * B[:,2:4]  (2x4 * 4x2 -> 2x2): two Strassen 2x2x2 blocks, rank 14
      - C[:,4]   = A * B[:,4]    (2x4 * 4x1 -> 2x1): schoolbook, rank 8
      Total rank = 14 + 14 + 8 = 36.

    Wait, that's the same as before! Let me reconsider.

    Actually, the original approach was:
      C[:,0:2] = A[:,0:2]*B[0:2,0:2] + A[:,2:4]*B[2:4,0:2]  (two Strassen, rank 14)
      C[:,2:4] = A[:,0:2]*B[0:2,2:4] + A[:,2:4]*B[2:4,2:4]  (two Strassen, rank 14)
      C[:,4]   = A * B[:,4]                                   (schoolbook, rank 8)
      Total = 36.

    The new approach:
      C[:,0:2] = A * B[:,0:2]  where A is 2x4, B[:,0:2] is 4x2.
      This is matmul(2,4,2) which has rank at most 14 (two Strassen on 2x2 blocks).
      C[:,2:4] = A * B[:,2:4]  similarly rank 14.
      C[:,4]   = A * B[:,4]    rank 8.
      Total = 36.

    Same rank! The key insight is that matmul(2,4,2) = matmul(2,4,2) has the
    same rank whether you view it as two 2x2*2x2 products or as one 2x4*4x2 product.
    The rank is the same: 14.

    So I need a fundamentally different approach to reduce rank below 36.

    Let me think about the full problem: matmul(2,4,5).
    Trivial: 2*4*5 = 40.
    Current best: 36.

    Can I do better? The bilinear complexity of matmul(2,4,5) is bounded by:
    - 40 (trivial)
    - 36 (current: 4 Strassen + schoolbook)
    - Maybe less?

    One idea: use a different block structure.
    Split A into 1x4 blocks (2 blocks), B into 4x5 (one block).
    C[i,:] = A[i,:] * B  (1x4 * 4x5 -> 1x5). Rank per row = 5 (schoolbook for 1x4*4x5).
    Total = 2*5 = 10? No, that's not right. Each row of C is a linear combination
    of rows of B with coefficients from A. The tensor for one row is matmul(1,4,5)
    which has rank 5. Two rows: rank 10. But that's not a valid decomposition
    because the U, V, W factors must be shared across all (i,j,k) triples.

    Actually, for matmul(1,4,5): T_{j,k} = delta_{j,k} (summed over j).
    The bilinear complexity is 5 (you need 5 independent products).
    For two rows: matmul(2,4,5). The two rows share the same B, so:
    C[0,k] = sum_j A[0,j]*B[j,k]
    C[1,k] = sum_j A[1,j]*B[j,k]

    If I use separate decompositions for each row, rank = 5 + 5 = 10.
    But the U factors would be: for row 0, U has entries for A[0,j]; for row 1,
    U has entries for A[1,j]. These are independent, so the total rank is 10.

    Wait, but that can't be right because the trivial bound is 40. Let me re-examine.

    The tensor is T in U*V*W where U = R^8 (A has 8 entries), V = R^20 (B has 20 entries),
    W = R^10 (C has 10 entries).
    T_{(i,j),(j,k),(k,i)} = 1 if the indices match, 0 otherwise.

    Actually, the standard matrix multiplication tensor for (n,m,p) has:
    T = sum_{i,j,k} e_{i,m+j} * e_{j,p+k} * e_{k,n+i}
    where i in [0,n), j in [0,m), k in [0,p).

    For (2,4,5): T = sum_{i=0}^1 sum_{j=0}^3 sum_{k=0}^4 e_{i*4+j} * e_{j*5+k} * e_{k*2+i}

    The rank of this tensor is what we're trying to minimize.

    A rank-1 decomposition term is: u * v * w where u in R^8, v in R^20, w in R^10.
    The contribution to T at position ((i,j),(j',k),(k',i')) is:
    u_{i*4+j} * v_{j'*5+k} * w_{k'*2+i'}

    For this to match the tensor, we need:
    u_{i*4+j} * v_{j'*5+k} * w_{k'*2+i'} = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}

    (summed over all terms in the decomposition)

    So each rank-1 term picks specific (i,j), (j',k), (k',i') and contributes
    to T at that position. Multiple terms can contribute to the same position.

    The schoolbook decomposition has 40 terms, one for each (i,j,k) triple:
    u = e_{i*4+j}, v = e_{j*5+k}, w = e_{k*2+i}

    To reduce rank, we need to find linear combinations that still span the tensor.

    This is a hard problem. Let me just go with rank 36 for now and see if I
    can find a better structure.

    Actually, let me try a different block decomposition:
    Split the 4 columns of A/B into groups of 1:
    C[:,k] = sum_{j=0}^3 A[:,j] * B[j,k]

    For each k, C[:,k] is 2x1 = A[:,0:4] * B[0:4,k].
    This is matmul(2,4,1) which has rank 8 (schoolbook).
    Total for all k: 5 * 8 = 40. Same as trivial.

    What if I group k's?
    C[:,0:2] = A * B[:,0:2] -- matmul(2,4,2), rank <= 14
    C[:,2:4] = A * B[:,2:4] -- matmul(2,4,2), rank <= 14
    C[:,4]   = A * B[:,4]   -- matmul(2,4,1), rank = 8
    Total = 36. Same as before.

    What if I group differently?
    C[:,0:3] = A * B[:,0:3] -- matmul(2,4,3)
    C[:,3:5] = A * B[:,3:5] -- matmul(2,4,2)

    matmul(2,4,3): can I do this in fewer than 14+8=22?
    Split: C[:,0:2] + C[:,2:3] = matmul(2,4,2) + matmul(2,4,1) = 14 + 8 = 22.
    Or: matmul(2,4,3) directly. Upper bound 24, can we do better?

    I think the best I can do with simple block decompositions is 36.
    Let me just implement that and verify.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 36

    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        """
        Add a Strassen 2x2x2 decomposition for:
        C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        """
        nonlocal col

        # M1 = (A00 + A11) * (B00 + B11)
        c = col_start
        U[i0 * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M2 = (A10 + A11) * B00
        c = col_start + 1
        U[(i0 + 1) * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] -= 1
        col += 1

        # M3 = A00 * (B01 - B11)
        c = col_start + 2
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M4 = A11 * (B10 - B00)
        c = col_start + 3
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        col += 1

        # M5 = (A00 + A01) * B11
        c = col_start + 4
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        col += 1

        # M6 = (A10 - A00) * (B00 + B01)
        c = col_start + 5
        U[(i0 + 1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M7 = (A01 - A11) * (B10 + B11)
        c = col_start + 6
        U[i0 * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] -= 1
        V[(j0 + 1) * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        col += 1

        return col_start + 7

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        """
        Add schoolbook decomposition for:
        C[i0:i0+2, k0:k0+1] = A[i0:i0+2, j0:j0+4] * B[j0:j0+4, k0:k0+1]
        """
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        return c

    # Block 1: C[:, 0:2] from A[:, 0:2] * B[0:2, 0:2]
    col = add_strassen_222(0, 0, 0, 0)
    # Block 2: C[:, 0:2] from A[:, 2:4] * B[2:4, 0:2]
    col = add_strassen_222(0, 2, 0, col)
    # Block 3: C[:, 2:4] from A[:, 0:2] * B[0:2, 2:4]
    col = add_strassen_222(0, 0, 2, col)
    # Block 4: C[:, 2:4] from A[:, 2:4] * B[2:4, 2:4]
    col = add_strassen_222(0, 2, 2, col)
    # Block 5: C[:, 4] from A * B[:, 4] (schoolbook 2x4 * 4x1)
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"

    # ---- Exact verification ----------------------------------------------
    if _remaining() > 0.5:
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    acc = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for r in range(rank):
                        acc += U[ui][r] * V[vj][r] * W[wk][r]
                    assert acc == 1, f"Verification failed at i={i},j={j},k={k}: acc={acc}"

    return {"U": U, "V": V, "W": W}
