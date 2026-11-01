import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6

    # Winograd Fourier Transform Algorithm:
    #   2x4x6 = (2x2x2) ⊗ (1x2x3)
    #   rank(2x2x2) = 7 (Strassen)
    #   rank(1x2x3) = 12
    #   Total rank = 7 * 12 = 84
    #
    # For the Kronecker product of two tensor decompositions:
    #   U_total = kron(U1, U2), V_total = kron(V1, V2), W_total = kron(W1, W2)
    #   with row ordering: row = r1 * dim_row2 + r2
    #   (i.e., first factor varies slowest)
    #
    # Factor 1: 2x2x2 Strassen
    #   U1 (4x7), V1 (4x7), W1 (4x7)
    #   A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    #   B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    #   C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    #
    # Strassen:
    #   P1 = (A00+A11)(B00+B11)
    #   P2 = (A10+A11)B00
    #   P3 = A00(B01-B11)
    #   P4 = A11(B10-B00)
    #   P5 = (A00+A01)B11
    #   P6 = (A10-A00)(B00+B01)
    #   P7 = (A01-A11)(B10+B11)
    #
    #   C00 = P1 + P4 - P5 + P7
    #   C01 = P3 + P5
    #   C10 = P2 + P4
    #   C11 = P1 - P2 + P3 + P6

    U1 = np.array([
        [ 1,  0,  1,  0,  1,  1,  0],   # A00
        [ 0,  0,  0,  0,  1,  0,  1],   # A01
        [ 0,  1,  0,  0,  0, -1,  0],   # A10
        [ 1,  1,  0,  1,  0,  0, -1],   # A11
    ], dtype=np.int64)  # (4, 7)

    V1 = np.array([
        [ 1,  1,  0,  0,  0,  1,  0],   # B00
        [ 0,  0,  1,  0,  0,  1,  0],   # B01
        [ 0,  0,  0,  1,  0,  0,  1],   # B10
        [ 1,  0, -1, -1,  1,  0,  1],   # B11
    ], dtype=np.int64)  # (4, 7)

    W1 = np.array([
        [ 1,  0,  0,  1, -1,  0,  1],   # C00
        [ 0,  0,  1,  0,  1,  0,  0],   # C01
        [ 0,  1,  0,  1,  0,  0,  0],   # C10
        [ 1, -1,  1,  0,  0,  1,  0],   # C11
    ], dtype=np.int64)  # (4, 7)

    # Factor 2: 1x2x3 rectangular multiplication
    #   A is 1x2: [a0, a1]  -> indices 0, 1
    #   B is 2x3: -> indices 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2)
    #   C is 1x3: [c0, c1, c2]  -> indices 0, 1, 2
    #
    #   c0 = a0*B00 + a1*B10
    #   c1 = a0*B01 + a1*B11
    #   c2 = a0*B02 + a1*B12
    #
    # Rank-12 decomposition (6 for a0 row + 6 for a1 row):
    #   Products P0..P5 involve a0 times B entries (rows 0,1,2 of B)
    #   Products P6..P11 involve a1 times B entries (rows 0,1,2 of B)
    #
    # This is the trivial (rank 6) but we need rank 12 for the
    # Kronecker structure to work with the 2x2x2 factor.
    #
    # Actually, for the Kronecker product to decompose 2x4x6 correctly:
    #   The 2x4x6 tensor = kron(2x2x2, 1x2x3)
    #   U_total has shape (2*2) x (7*12) = 4x84  ... wait
    #
    # Let me reconsider. The Kronecker product of tensors:
    #   T1 in R^{2x2x2} (Strassen, rank 7)
    #   T2 in R^{1x2x3} (rank r2)
    #   T1⊗T2 in R^{2x4x6} (rank 7*r2)
    #
    # The decomposition:
    #   U_total[i1*dim_U2 + i2, r1*rank2 + r2_idx] = U1[i1,r1] * U2[i2,r2_idx]
    #   V_total[j1*dim_V2 + j2, r1*rank2 + r2_idx] = V1[j1,r1] * V2[j2,r2_idx]
    #   W_total[k1*dim_W2 + k2, r1*rank2 + r2_idx] = W1[k1,r1] * W2[k2,r2_idx]
    #
    # Where for 2x4x6:
    #   U: A is 2x4, so U has 8 rows. Row = i*4+j where i in {0,1}, j in {0,1,2,3}
    #   In Kronecker: row = i1 * (1*2) + (i2*2 + j2) where i1 in {0,1} (from 2x2x2),
    #                  i2 in {0} (from 1x2x3, only 1 row), j2 in {0,1} (from 2 cols of 1x2x3)
    #   Wait, this doesn't map cleanly.
    #
    # Let me think more carefully.
    #
    # 2x4x6 = (2x2x2) ⊗ (1x2x3):
    #   2 = 2*1, 4 = 2*2, 6 = 2*3
    #
    # A (2x4): element A[i, j] where i in {0,1}, j in {0,1,2,3}
    #   In Kronecker: i = i1*1 + i2 where i1 in {0,1}, i2 in {0} (since 1 dim)
    #                 j = j1*2 + j2 where j1 in {0,1}, j2 in {0,1}
    #   So A[i1,i2][j1,j2] = A1[i1,j1] * A2[i2,j2] but A2 is 1x2...
    #
    # Actually the Kronecker product of matrices:
    #   A (2x4) = A1 (2x2) ⊗ A2 (1x2)
    #   A[i,j] = A1[i1,j1] * A2[i2,j2] where i = i1*1 + i2, j = j1*2 + j2
    #
    #   B (4x6) = B1 (2x2) ⊗ B2 (2x3)
    #   B[j,k] = B1[j1,k1] * B2[j2,k2] where j = j1*2 + j2, k = k1*3 + k2
    #
    #   C (2x6) = C1 (2x2) ⊗ C2 (1x3)
    #   C[i,k] = C1[i1,k1] * C2[i2,k2] where i = i1*1 + i2, k = k1*3 + k2
    #
    # The multiplication tensor: C[i,k] = sum_{j} A[i,j] * B[j,k]
    # In Kronecker: C1[i1,k1]*C2[i2,k2] = sum_{j1,j2} (A1[i1,j1]*A2[i2,j2]) * (B1[j1,k1]*B2[j2,k2])
    #   = (sum_{j1} A1[i1,j1]*B1[j1,k1]) * (sum_{j2} A2[i2,j2]*B2[j2,k2])
    #   = C1[i1,k1] * C2[i2,k2]  ✓
    #
    # So the decomposition factors:
    #   U_total[i, r] where i = i1*1 + i2 = i1 + i2, r = r1*rank2 + r2
    #     U_total[i1 + i2, r1*rank2 + r2] = U1[i1, r1] * U2[i2, r2]
    #     But i1 in {0,1}, i2 in {0}, so i = i1 (i2 is always 0)
    #     Wait, that gives only 2 rows for U, but we need 8.
    #
    # I think I'm confusing the Kronecker structure. Let me use the standard approach:
    #
    # For the Kronecker product of two tensor decompositions (T1 of rank r1, T2 of rank r2):
    #   The combined tensor has rank r1*r2.
    #   U_combined = kron(U1, U2)  (standard Kronecker product of matrices)
    #   V_combined = kron(V1, V2)
    #   W_combined = kron(W1, W2)
    #
    #   Where kron(U1, U2) has shape (rows1*rows2, cols1*cols2)
    #   and (kron(U1,U2))[i1*rows2 + i2, j1*cols2 + j2] = U1[i1,j1] * U2[i2,j2]
    #
    # For our case:
    #   U1 is 4x7 (for 2x2x2: A has 4 entries)
    #   U2 is 2x12 (for 1x2x3: A has 2 entries)
    #   U_combined = kron(U1, U2) is (4*2) x (7*12) = 8 x 84  ✓
    #
    #   V1 is 4x7 (B has 4 entries)
    #   V2 is 6x12 (B has 6 entries)
    #   V_combined = kron(V1, V2) is (4*6) x (7*12) = 24 x 84  ✓
    #
    #   W1 is 4x7 (C has 4 entries)
    #   W2 is 3x12 (C has 3 entries)
    #   W_combined = kron(W1, W2) is (4*3) x (7*12) = 12 x 84  ✓

    # For 1x2x3, the trivial decomposition has rank 6, but we need to verify
    # that kron with Strassen gives a valid decomposition.
    # Actually, ANY valid decomposition of 1x2x3 works. Let's use rank 6:
    #
    #   c0 = a0*B00 + a1*B10  -> 2 products
    #   c1 = a0*B01 + a1*B11  -> 2 products
    #   c2 = a0*B02 + a1*B12  -> 2 products
    #   Total rank = 6
    #
    # But wait, can we do better than 6 for 1x2x3?
    #   The 1x2x3 tensor has rank... it's a simple multiplication.
    #   a0*B00, a0*B01, a0*B02, a1*B10, a1*B11, a1*B12
    #   These are 6 rank-1 terms. Can we combine any?
    #   The rank of the 1x2x3 multiplication tensor is 6 (it's an identity-like map).
    #
    # So total rank = 7 * 6 = 42. Same as before!
    #
    # To get below 42, we need a better decomposition of one factor.
    # But the Kronecker structure locks us into 7*6=42 if both are minimal.
    #
    # Alternative: don't use Kronecker product directly. Use a smarter block approach.
    #
    # Let me try: split 2x4x6 as 2x(2+2)x(2+2+2) and use Strassen for each 2x2x2 block
    # but share some computations across blocks.
    #
    # Actually, the current program already does this and gets rank 42.
    #
    # To improve, I need a fundamentally different approach.
    #
    # Let me try the rank-36 approach: use a non-Kronecker decomposition.
    #
    # 2x4x6: A(2x4) @ B(4x6) = C(2x6)
    #
    # Partition B into two 4x3 blocks: B = [B1 | B2] where B1, B2 are 4x3.
    # C = [A@B1 | A@B2]
    #
    # If I can compute A@B1 (2x4 @ 4x3) in rank r and A@B2 in rank r, total = 2r.
    # Rank of 2x4x3: known to be 12 (from literature).
    # So total = 24? That seems too good. Let me verify.
    #
    # Actually, the rank of 2x4x3 is not simply 12. Let me think.
    # 2x4x3: A(2x4), B(4x3), C(2x3)
    #   C[0,0] = a00*b00 + a01*b10 + a02*b20 + a03*b30
    #   C[0,1] = a00*b01 + a01*b11 + a02*b21 + a03*b31
    #   C[0,2] = a00*b02 + a01*b12 + a02*b22 + a03*b32
    #   C[1,0] = a10*b00 + a11*b10 + a12*b20 + a13*b30
    #   C[1,1] = a10*b01 + a11*b11 + a12*b21 + a13*b31
    #   C[1,2] = a10*b02 + a11*b12 + a12*b22 + a13*b32
    #
    # This is essentially two independent 1x4x3 multiplications (one for each row of A).
    # Rank of 1x4x3 = 4 (trivially: a0*b0, a1*b1, a2*b2, a3*b3 for each output).
    # Wait no, 1x4x3: c0 = a0*b0 + a1*b1 + a2*b2 + a3*b3 (for each column)
    # That's 3 outputs, each requiring 4 products, but the a's are shared.
    # Rank of 1x4x3 is actually 4 (you need at least 4 products for 4 input a's).
    #
    # So 2x4x3 = two copies of 1x4x3, rank = 2*4 = 8? No, that's not right either
    # because the two rows of A are independent.
    #
    # Actually, 2x4x3 has rank 12. Here's why:
    #   The tensor is a direct sum of two 1x4x3 tensors (one for each row of A).
    #   Each 1x4x3 has rank 4 (you need 4 multiplications: a0*b0, a1*b1, a2*b2, a3*b3
    #   for each of the 3 outputs... no that's 12).
    #
    # Hmm, let me just count: for 1x4x3, you have 4 a's and 3 b's per column, 3 columns.
    # The minimal rank is 4 (one per a, combining all b's).
    # Wait: c_k = sum_j a_j * b_jk for k=0,1,2.
    # If we set p_j = a_j * (b_j0, b_j1, b_j2) - that's not a single product.
    #
    # For rank-1: c_k = (sum of a_j * u_j) * (sum of b_jk * v_jk)
    # This requires the b matrix to be rank 1, which it's not.
    #
    # The rank of the 1xn xm multiplication tensor is n*m (I think).
    # So 1x4x3 has rank 12? That can't be right.
    #
    # Let me just use the trivial decomposition: rank 12 for 1x4x3.
    # For each (j,k) pair, one product: a_j * b_jk.
    # That's 4*3 = 12 products.
    #
    # So 2x4x3 has rank 24 (two independent 1x4x3's).
    # And 2x4x6 = 2x4x3 + 2x4x3 has rank 48.
    # That's worse than 42!
    #
    # The Strassen block approach (rank 42) is better because it exploits the 2x2 structure.
    #
    # OK so rank 42 is likely close to optimal for 2x4x6.
    # Let me just return the rank-42 solution to guarantee validity.

    # Rank 42 solution using Strassen blocks.
    rank = 42

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    strassen_U = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1},
    ]

    strassen_V = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]

    strassen_W = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]

    col = 0
    for j_block in range(2):
        for k_block in range(3):
            for prod_idx in range(7):
                r = col + prod_idx

                for local_idx, sign in strassen_U[prod_idx].items():
                    gi = local_idx // 2
                    gj = local_idx % 2
                    U[gi * m + (j_block * 2 + gj), r] = sign

                for local_idx, sign in strassen_V[prod_idx].items():
                    gj = local_idx // 2
                    gk = local_idx % 2
                    V[(j_block * 2 + gj) * p + (k_block * 2 + gk), r] = sign

                for local_idx, sign in strassen_W[prod_idx].items():
                    gi = local_idx // 2
                    gk = local_idx % 2
                    W[(k_block * 2 + gk) * n + gi, r] = sign

            col += 7

    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
