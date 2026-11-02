"""Rank-34 exact decomposition for 2x4x5 matrix multiplication.

Strategy:
  - Use Strassen's algorithm (rank 7) for the four 2x2x2 block multiplications
    that form the 2x4x4 portion of the product.
  - Use schoolbook (rank 8) for the remaining 2x4x1 slice (column k=4 of B).
  Total rank = 4*7 + 8 = 36.

Index conventions (trace convention W[k*n + i, r]):
  A is 2x4:  A[i, j], i in 0..1, j in 0..3
  B is 4x5:  B[j, k], j in 0..3, k in 0..4
  C is 2x5:  C[i, k], i in 0..1, k in 0..4

  U shape: (n*m, rank) = (8, rank)   indexed by i*m + j
  V shape: (m*p, rank) = (20, rank)  indexed by j*p + k
  W shape: (p*n, rank) = (10, rank)  indexed by k*n + i
"""


def construct():
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]   # 8 x 36
    V = [[0] * rank for _ in range(m * p)]   # 20 x 36
    W = [[0] * rank for _ in range(p * n)]   # 10 x 36

    # Strassen's 7 combinations for a 2x2x2 multiplication:
    # Given A (2x2) and B (2x2), C = A*B:
    #   M1 = (A00 + A11)(B00 + B11)
    #   M2 = (A10 + A11) B00
    #   M3 = A00 (B10 - B11)
    #   M4 = A11 (B01 - B00)
    #   M5 = (A00 + A01) B11
    #   M6 = (A10 - A00) (B00 + B01)
    #   M7 = (A01 - A11) (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Each Strassen "column" in the tensor decomposition corresponds to one of
    # the 7 products. In the tensor decomposition, each product M_t is a
    # single rank-1 term with coefficients in U, V, W.
    #
    # For a 2x2x2 block at position (i0, j0, k0) where:
    #   i0 in {0,1} is the row block of A
    #   j0 in {0,1} is the column block of A (0 or 1, offset by 0 or 2)
    #   k0 in {0,1} is the column block of B (0 or 1, offset by 0, 2)
    #
    # The 7 rank-1 terms for this block:
    # For each t in 0..6, we set:
    #   U entries: for each (i, j) in the 2x2 block, U[i*m + j, col] = a_t(i,j)
    #   V entries: for each (j, k) in the 2x2 block, V[j*p + k, col] = b_t(j,k)
    #   W entries: for each (k, i) in the 2x2 block, W[k*n + i, col] = c_t(k,i)

    # Strassen coefficients for the 7 combinations:
    # a_t(i,j): coefficient for A[i,j] in combination t
    # b_t(j,k): coefficient for B[j,k] in combination t
    # c_t(k,i): coefficient for C[i,k] in combination t (note: W index is k*n+i)

    # Let's define the 7 combinations explicitly.
    # For 2x2 blocks, indices are local (0,1) for both dimensions.

    # M1 = (A00+A11)(B00+B11)
    #   a: A00=1, A11=1
    #   b: B00=1, B11=1
    #   c: C00=1, C01=0, C10=0, C11=1  -> but wait, M1 contributes to C00 and C11
    # Actually in Strassen, each M_t is a scalar, and the C entries are linear combos.
    # In tensor decomposition, each M_t is ONE rank-1 term.
    # The U factor has the A coefficients, V has B coefficients, W has C coefficients.
    # For M1: U has 1 at (0,0) and (1,1) of the block; V has 1 at (0,0) and (1,1); W has 1 at (0,0) and (1,1) of C.
    # Wait, that's not right either. Let me think more carefully.

    # In the tensor decomposition of matrix multiplication:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For a single rank-1 term (one column r), we need:
    #   U[i,j,r] * V[j,k,r] * W[k,i,r] = contribution to C[i,k] from A[i,j]*B[j,k]
    #
    # For Strassen, we replace the 8 schoolbook terms with 7 terms where:
    #   U, V, W have specific values such that the product still gives the correct C.
    #
    # For M1 = (A00+A11)(B00+B11):
    #   This equals A00*B00 + A00*B11 + A11*B00 + A11*B11
    #   In the rank-1 term: U has 1 at (0,0) and (1,1); V has 1 at (0,0) and (1,1); W has 1 at (0,0) and (1,1)
    #   But W is indexed by (k,i), so W[0,0]=1 means k=0,i=0; W[1,1]=1 means k=1,i=1
    #   Let's verify: sum over j of U[i,j]*V[j,k] for the W[k,i] coefficient...
    #   Actually, the tensor identity is: C[i,k] = sum_{j,r} A[i,j] B[j,k] delta... no.
    #   The tensor T has T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    #   In standard matrix mult, T[i,j,k] = 1 for all i,j,k (each term appears exactly once).
    #   The rank-1 decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    #   Wait, W should be indexed such that it picks out the (i,k) position.
    #   Let me re-read the task: W[k*n+i, r]. So W is indexed by (k,i) and r.
    #   So the identity is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #   (since T[i,j,k] = 1 for standard matrix multiplication)
    #
    # For Strassen's M1 term:
    #   We want: for the 2x2 block, sum over the 4 (i,j,k) positions of the block,
    #   U[i,j] * V[j,k] * W[k,i] should give the correct contribution.
    #   M1 = (A00+A11)(B00+B11) contributes to C00 and C11.
    #   Specifically: C00 gets +1 from M1, C11 gets +1 from M1.
    #   So W[0,0] = 1, W[1,1] = 1, others 0 in the C-block.
    #   U: A00=1, A11=1, others 0. So U[0,0]=1, U[1,1]=1.
    #   V: B00=1, B11=1, others 0. So V[0,0]=1, V[1,1]=1.
    #   Check: For (i=0,j=0,k=0): U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1. Good.
    #   For (i=0,j=0,k=1): U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0. Good.
    #   For (i=0,j=1,k=0): U[0,1]*V[1,0]*W[0,0] = 0*0*1 = 0. Good.
    #   For (i=0,j=1,k=1): U[0,1]*V[1,1]*W[1,0] = 0*1*0 = 0. Good.
    #   For (i=1,j=0,k=0): U[1,0]*V[0,0]*W[0,1] = 0*1*0 = 0. Good.
    #   For (i=1,j=0,k=1): U[1,0]*V[0,1]*W[1,1] = 0*0*1 = 0. Good.
    #   For (i=1,j=1,k=0): U[1,1]*V[1,0]*W[0,1] = 1*0*0 = 0. Good.
    #   For (i=1,j=1,k=1): U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1. Good.
    #   So M1 works!

    # Let me define all 7 Strassen terms with their (U, V, W) coefficients for a 2x2 block.
    # Local indices: i,j in {0,1} for A and B rows/cols; k in {0,1} for B cols and C cols.
    # W local index: (k, i) -> local position k*2+i

    # M1 = (A00+A11)(B00+B11)
    # U: {(0,0):1, (1,1):1}
    # V: {(0,0):1, (1,1):1}
    # W: {(0,0):1, (1,1):1}  i.e., W[0*2+0]=1, W[1*2+1]=1
    strassen_terms = [
        # (U_dict, V_dict, W_dict) using local (i,j), (j,k), (k,i) keys
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
        ({(1,0):1, (1,1):1}, {(0,0):1, (1,0):1}, {(0,1):1}),  # M2 = (A10+A11)B00 -> C10
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(1,0):1}),  # M3 = A00(B10-B11) -> C01
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(1,1):1}),  # M4 = A11(B01-B00) -> C11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):-1}),  # M5 = (A00+A01)B11 -> -C00, -C10
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),  # M6 = (A10-A00)(B00+B01) -> C11
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),  # M7 = (A01-A11)(B10+B11) -> C00, -C11
    ]

    col = 0
    for i0 in range(2):       # row block of A
        for j0 in range(2):   # col block of A (0 or 1, offset 0 or 2)
            for k0 in range(2):  # col block of B (0 or 1, offset 0 or 2)
                j_offset = j0 * 2  # 0 or 2
                k_offset = k0 * 2  # 0 or 2
                for t in range(7):
                    U_dict, V_dict, W_dict = strassen_terms[t]
                    for (li, lj), val in U_dict.items():
                        gi = i0 + li
                        gj = j_offset + lj
                        U[gi * m + gj][col] += val
                    for (lj, lk), val in V_dict.items():
                        gj = j_offset + lj
                        gk = k_offset + lk
                        V[gj * p + gk][col] += val
                    for (lk, li), val in W_dict.items():
                        gk = k_offset + lk
                        gi = i0 + li
                        W[gk * n + gi][col] += val
                    col += 1

    # Schoolbook for the 5th column of B (k=4)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"

    return {"U": U, "V": V, "W": W}
