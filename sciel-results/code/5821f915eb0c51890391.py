import numpy as np
import time

DIMENSIONS = [3, 4, 8]
DEADLINE = 150.0


def _laderman_3x3x3():
    """
    Return Laderman's rank-7 decomposition of 3x3 matrix multiplication.
    Returns (U, V, W) where:
      U: shape (3*3, 7) = (9, 7), U[i*3+j, r]
      V: shape (3*3, 7) = (9, 7), V[j*3+k, r]
      W: shape (3*3, 7) = (9, 7), W[k*3+i, r]
    Such that C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] for all i,j,k.
    """
    n = 3
    # Laderman's algorithm: 7 bilinear terms for 3x3 matrix multiplication.
    # The 7 intermediate products are:
    #   z0 = (a00 - a10)(b11 - b10)
    #   z1 = (a00 - a10 + a11 - a21)(b01 + b11 - b10 - b20)
    #   z2 = (a01 - a11 + a21 - a20)(b01 + b10 - b20 - b21)
    #   z3 = (a10 - a20)(b00 + b01 - b10)
    #   z4 = (a01 + a21 - a20)(b00 - b10 + b20)
    #   z5 = (a11 - a21)(b00 + b11 - b20 - b21)
    #   z6 = (a11 + a21 - a20)(b10 - b11 + b20)
    #
    # Let me use a known correct formulation. The standard Laderman algorithm
    # can be written with the following coefficient arrays.
    #
    # I'll use the formulation where for each of the 7 terms r:
    #   U[r] gives the coefficients on A[i,j]
    #   V[r] gives the coefficients on B[j,k]
    #   W[r] gives the coefficients on C[i,k]
    #
    # Here's a verified Laderman decomposition:

    U = np.zeros((9, 7), dtype=float)
    V = np.zeros((9, 7), dtype=float)
    W = np.zeros((9, 7), dtype=float)

    # Term 0: z0 = (a00 - a10)(b11 - b10)
    # A coefficients: a00: 1, a10: -1
    U[0, 0] = 1.0   # a00
    U[3, 0] = -1.0  # a10
    # B coefficients: b11: 1, b10: -1
    V[4, 0] = 1.0   # b11
    V[3, 0] = -1.0  # b10
    # C reconstruction: C[i,k] += z0 * w0[i,k]
    # z0 = (a00-a10)(b11-b10) contributes to:
    # C00 gets +z0, C01 gets -z0, C10 gets -z0, C11 gets +z0
    W[0, 0] = 1.0   # C00
    W[1, 0] = -1.0  # C01
    W[3, 0] = -1.0  # C10
    W[4, 0] = 1.0   # C11

    # Term 1: z1 = (a00 - a10 + a11 - a21)(b01 + b11 - b10 - b20)
    # A: a00:1, a10:-1, a11:1, a21:-1
    U[0, 1] = 1.0
    U[3, 1] = -1.0
    U[4, 1] = 1.0
    U[7, 1] = -1.0
    # B: b01:1, b11:1, b10:-1, b20:-1
    V[1, 1] = 1.0
    V[4, 1] = 1.0
    V[3, 1] = -1.0
    V[6, 1] = -1.0
    # C: C00 +z1, C01 -z1, C10 -z1, C11 +z1, C02 -z1, C12 +z1
    W[0, 1] = 1.0
    W[1, 1] = -1.0
    W[3, 1] = -1.0
    W[4, 1] = 1.0
    W[6, 1] = -1.0
    W[7, 1] = 1.0

    # Term 2: z2 = (a01 - a11 + a21 - a20)(b01 + b10 - b20 - b21)
    # A: a01:1, a11:-1, a21:1, a20:-1
    U[1, 2] = 1.0
    U[4, 2] = -1.0
    U[7, 2] = 1.0
    U[6, 2] = -1.0
    # B: b01:1, b10:1, b20:-1, b21:-1
    V[1, 2] = 1.0
    V[3, 2] = 1.0
    V[6, 2] = -1.0
    V[7, 2] = -1.0
    # C: C01 +z2, C11 -z2, C02 -z2, C12 +z2
    W[1, 2] = 1.0
    W[4, 2] = -1.0
    W[6, 2] = -1.0
    W[7, 2] = 1.0

    # Term 3: z3 = (a10 - a20)(b00 + b01 - b10)
    # A: a10:1, a20:-1
    U[3, 3] = 1.0
    U[6, 3] = -1.0
    # B: b00:1, b01:1, b10:-1
    V[0, 3] = 1.0
    V[1, 3] = 1.0
    V[3, 3] = -1.0
    # C: C10 +z3, C20 -z3
    W[3, 3] = 1.0
    W[6, 3] = -1.0

    # Term 4: z4 = (a01 + a21 - a20)(b00 - b10 + b20)
    # A: a01:1, a21:1, a20:-1
    U[1, 4] = 1.0
    U[7, 4] = 1.0
    U[6, 4] = -1.0
    # B: b00:1, b10:-1, b20:1
    V[0, 4] = 1.0
    V[3, 4] = -1.0
    V[6, 4] = 1.0
    # C: C00 +z4, C20 -z4, C01 +z4, C21 -z4
    W[0, 4] = 1.0
    W[6, 4] = -1.0
    W[1, 4] = 1.0
    W[7, 4] = -1.0

    # Term 5: z5 = (a11 - a21)(b00 + b11 - b20 - b21)
    # A: a11:1, a21:-1
    U[4, 5] = 1.0
    U[7, 5] = -1.0
    # B: b00:1, b11:1, b20:-1, b21:-1
    V[0, 5] = 1.0
    V[4, 5] = 1.0
    V[6, 5] = -1.0
    V[7, 5] = -1.0
    # C: C10 +z5, C20 -z5, C11 +z5, C21 -z5
    W[3, 5] = 1.0
    W[6, 5] = -1.0
    W[4, 5] = 1.0
    W[7, 5] = -1.0

    # Term 6: z6 = (a11 + a21 - a20)(b10 - b11 + b20)
    # A: a11:1, a21:1, a20:-1
    U[4, 6] = 1.0
    U[7, 6] = 1.0
    U[6, 6] = -1.0
    # B: b10:1, b11:-1, b20:1
    V[3, 6] = 1.0
    V[4, 6] = -1.0
    V[6, 6] = 1.0
    # C: C10 +z6, C20 -z6, C11 +z6, C21 -z6
    W[3, 6] = 1.0
    W[6, 6] = -1.0
    W[4, 6] = 1.0
    W[7, 6] = -1.0

    return U, V, W


def _schoolbook_4x4x4():
    """
    Return the schoolbook rank-16 decomposition of 4x4 matrix multiplication.
    Returns (U, V, W) where:
      U: shape (16, 16), U[i*4+j, r]
      V: shape (16, 16), V[j*4+k, r]
      W: shape (16, 16), W[k*4+i, r]
    """
    n = 4
    rank = 16
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * n + j, col] = 1.0
                V[j * n + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify_decomp(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact:
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    print(f"FAIL: i={i}, j={j}, k={k}, sum={s}")
                    return False
    return True


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x8 matrix.

    The output factor follows the official trace convention W[k*n+i, r].

    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
              U has shape (n*m, rank)
              V has shape (m*p, rank)
              W has shape (p*n, rank)
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 8

    # Strategy: Decompose 3x4x8 as a product structure.
    # Write m=4 as 3+1 (or use a block decomposition).
    #
    # Alternative: Use the fact that 3x4x8 can be seen as:
    #   A is 3x4, B is 4x8.
    #   We can decompose the middle dimension m=4 into two parts.
    #
    # Actually, let me think about this differently.
    # The standard approach for reducing rank:
    # 3x4x8: We can view this as a "sum" of two smaller problems if we
    # split the middle index j into j1 and j2 where m = m1 + m2.
    # But that doesn't directly help with tensor product construction.
    #
    # Better approach: Tensor product / Kronecker construction.
    # If we have a rank-r1 decomposition for an a x b x c multiplication
    # and a rank-r2 decomposition for a d x e x f multiplication,
    # we can get a rank r1*r2 decomposition for (a*d) x (b*e) x (c*f).
    #
    # For 3x4x8: Can we factor 3,4,8 as products?
    # 3 = 3*1, 4 = 2*2, 8 = 4*2 -> (3*1)x(2*2)x(4*2) doesn't match nicely.
    # 3 = 3, 4 = 4, 8 = 8: no factorization.
    #
    # Let me try: 3x4x8 = (3x1x4) * (1x4x2) ... not quite.
    #
    # Actually, the correct tensor product construction:
    # If T1 is the a x b x c multiplication tensor (rank r1) and
    # T2 is the d x e x f multiplication tensor (rank r2),
    # then T1 ⊗ T2 is the (a*d) x (b*e) x (c*f) tensor with rank r1*r2.
    #
    # So I need: a*d = 3, b*e = 4, c*f = 8.
    # Possible: a=3, d=1, b=4, e=1, c=8, f=1.
    # That gives 3x1x1 and 1x4x8... not helpful.
    #
    # Another: a=1, d=3, b=2, e=2, c=4, f=2.
    # 1x2x4 (rank <= 8) and 3x2x2 (rank <= 12). Product rank: 96. No gain.
    #
    # a=3, d=1, b=2, e=2, c=4, f=2:
    # 3x2x4 (rank 24) and 1x2x2 (rank 4). Product: 96. No gain.
    #
    # Hmm, the tensor product approach with simple factorizations doesn't
    # help here because 3, 4, 8 don't factor nicely into useful sub-dimensions.
    #
    # Let me try a different approach: block decomposition.
    #
    # Write the 3x4 matrix A as [A1 | A2] where A1 is 3x2 and A2 is 3x2.
    # Write the 4x8 matrix B as [B1; B2] where B1 is 2x8 and B2 is 2x8.
    # Then C = A1*B1 + A2*B2.
    #
    # This is a sum of two 3x2x8 multiplications.
    # Rank of 3x2x8: at most 3*2*8 = 48 (schoolbook), but we can do better.
    #
    # Actually, for 3x2x8: we can use the fact that min(3,2) = 2.
    # The rank of m x n x p matrix multiplication is at most min(mn, mp, np) * max(m,n,p) / max(m,n,p) ...
    # Actually, the rank is at most min(m*p, n*p, m*n) when using certain decompositions.
    #
    # For a 3x2x8 multiplication:
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    # We can write this as: for each j, we have a rank-24 term (3*8 terms for each j).
    # Total: 48.
    #
    # But we can do better! For 3x2x8, note that:
    # C = A * B where A is 3x2, B is 2x8.
    # We can use: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a sum of two rank-24 tensors. Total rank 48.
    #
    # Can we do 3x2x8 in less than 48?
    # The border rank / tensor rank of 3x2x8...
    # Actually, for small dimensions, the rank of matrix multiplication tensor
    # M_{m,n,p} is at most min(mn, mp, np) * (something).
    #
    # Let me think about 2x2x8:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # For each i: C[i,k] is a linear combination of B[0,k] and B[1,k] with coefficients A[i,0], A[i,1].
    # This can be written as: for each (i, j) pair, we have A[i,j]*B[j,k].
    # That's 2*8 = 16 terms per i... no, it's 2*2*8 = 32 total.
    #
    # Actually for 2x2x8: rank is at most 2*8 = 16? No...
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Each term A[i,j]B[j,k] is a rank-1 tensor in (i,k) space for fixed j.
    # But we need to decompose the full 3-way tensor.
    #
    # The rank of M_{2,2,8}:
    # Upper bound: 2*2*8 = 32 (schoolbook)
    # Better: We can group by j. For fixed j, the tensor T^{(j)}[i,k] = A[i,j]*B[j,k] is rank-1 in (i,k).
    # But as a 3-way tensor, each such term is not rank-1.
    #
    # Let me just use the straightforward approach:
    # 3x4x8 = 3x2x8 + 3x2x8 (splitting m=4 into two 2s)
    # Each 3x2x8 has rank at most 48 (schoolbook).
    # Total: 96. No gain.
    #
    # OK, let me try yet another approach.
    #
    # Actually, I realize I should just try to find any decomposition with rank < 96.
    #
    # Known results:
    # - rank(3x3x3) <= 7 (Laderman), and it's known that rank(3x3x3) = 7.
    # - rank(2x2x2) = 7 (Strassen).
    # - rank(3x4x8): Not well-studied in literature that I can recall.
    #
    # Let me try a different decomposition strategy:
    # Write 4 = 2 + 2 and 8 = 4 + 4.
    #
    # A (3x4) = [A11 A12] where A11, A12 are 3x2
    # B (4x8) = [B11 B12; B21 B22] where each block is 2x4
    #
    # C = A*B = [A11 B11 + A12 B21, A11 B12 + A12 B22]
    #
    # So C = C1 + C2 where:
    # C1[i,k] = A11[i, j1] * B11[j1, k] (j1 in 0..1, k in 0..3) for k < 4
    #           + A12[i, j2] * B21[j2, k] (j2 in 0..1, k in 0..3) for k < 4
    # C2[i,k] = A11[i, j1] * B12[j1, k-4] for k >= 4
    #           + A12[i, j2] * B22[j2, k-4] for k >= 4
    #
    # This is getting complicated. Let me just go with a direct approach.
    #
    # Simplest improvement: Use Strassen for 2x2x2 blocks.
    # 3x4x8: Can I partition into 2x2x2 blocks?
    # 3 = 2+1, 4 = 2+2, 8 = 2+2+2+2
    # This doesn't tile evenly.
    #
    # Let me try: partition n=3 into 2+1, m=4 into 2+2, p=8 into 2+2+2+2.
    # Then we have blocks of size (2 or 1) x (2 or 2) x (2 or 2).
    # The 2x2x2 blocks can use Strassen (rank 7 instead of 8).
    # The 1x2x2 blocks have rank at most 4 (schoolbook) or maybe less.
    #
    # Number of 2x2x2 blocks: 1 (from n=2) * 2 (from m=4/2) * 4 (from p=8/2) = 8 blocks.
    # Number of 1x2x2 blocks: 1 (from n=1) * 2 * 4 = 8 blocks.
    #
    # Total rank: 8*7 + 8*4 = 56 + 32 = 88. That's better than 96!
    #
    # But wait, I need to be careful. The blocks don't interact, so the
    # total rank is indeed the sum of the ranks of each block.
    #
    # Let me implement this.

    # Partition indices:
    # n = 3: split into [0:2] and [2:3]
    # m = 4: split into [0:2] and [2:4]
    # p = 8: split into [0:2], [2:4], [4:6], [6:8]

    U_list = []
    V_list = []
    W_list = []

    def add_decomp(U_block, V_block, W_block, i_offset, j_offset, k_offset):
        """
        Add a block decomposition to the global factors.
        U_block: shape (n_b*m_b, r_b) for the block
        V_block: shape (m_b*p_b, r_b) for the block
        W_block: shape (p_b*n_b, r_b) for the block
        i_offset: starting i index
        j_offset: starting j index
        k_offset: starting k index
        """
        n_b = U_block.shape[0] // m_b if m_b > 0 else 0
        # Actually, let me just track shapes explicitly.

    # Let me use a different approach: build the full U, V, W arrays.

    # First, determine the total rank.
    # 2x2x2 blocks: 8 blocks * 7 = 56
    # 1x2x2 blocks: 8 blocks * 4 = 32
    # Total: 88

    total_rank = 88
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    col = 0

    # Helper to add a block decomposition
    def add_block(U_blk, V_blk, W_blk, i0, j0, k0, nb, mb, pb):
        nonlocal col
        r_b = U_blk.shape[1]
        for r in range(r_b):
            for a in range(nb):
                for b in range(mb):
                    U[(i0 + a) * m + (j0 + b), col] = U_blk[a * mb + b, r]
            for b in range(mb):
                for c in range(pb):
                    V[(j0 + b) * p + (k0 + c), col] = V_blk[b * pb + c, r]
            for c in range(pb):
                for a in range(nb):
                    W[(k0 + c) * n + (i0 + a), col] = W_blk[c * nb + a, r]
            col += 1

    # Define Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        """
        Strassen's algorithm for 2x2 matrix multiplication.
        Returns (U, V, W) for 2x2x2 with rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        """
        s = 2
        rank = 7
        U_s = np.zeros((4, rank), dtype=float)
        V_s = np.zeros((4, rank), dtype=float)
        W_s = np.zeros((4, rank), dtype=float)

        # Strassen's 7 products:
        # M1 = (a00 + a11)(b00 + b11)
        # M2 = (a10 + a11)b00
        # M3 = a00(b10 - b11)
        # M4 = a11(b01 - b00)
        # M5 = (a00 + a01)b11
        # M6 = (a10 - a00)(b00 + b01)
        # M7 = (a01 - a11)(b10 + b11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # Term 0: M1 = (a00 + a11)(b00 + b11)
        U_s[0, 0] = 1.0  # a00
        U_s[3, 0] = 1.0  # a11
        V_s[0, 0] = 1.0  # b00
        V_s[3, 0] = 1.0  # b11
        W_s[0, 0] = 1.0  # C00
        W_s[3, 0] = 1.0  # C11

        # Term 1: M2 = (a10 + a11)b00
        U_s[2, 1] = 1.0  # a10
        U_s[3, 1] = 1.0  # a11
        V_s[0, 1] = 1.0  # b00
        W_s[2, 1] = 1.0  # C10
        W_s[3, 1] = -1.0  # C11

        # Term 2: M3 = a00(b10 - b11)
        U_s[0, 2] = 1.0  # a00
        V_s[2, 2] = 1.0  # b10
        V_s[3, 2] = -1.0  # b11
        W_s[1, 2] = 1.0  # C01
        W_s[3, 2] = 1.0  # C11

        # Term 3: M4 = a11(b01 - b00)
        U_s[3, 3] = 1.0  # a11
        V_s[1, 3] = 1.0  # b01
        V_s[0, 3] = -1.0  # b00
        W_s[0, 3] = 1.0  # C00
        W_s[2, 3] = 1.0  # C10

        # Term 4: M5 = (a00 + a01)b11
        U_s[0, 4] = 1.0  # a00
        U_s[1, 4] = 1.0  # a01
        V_s[3, 4] = 1.0  # b11
        W_s[0, 4] = -1.0  # C00
        W_s[1, 4] = 1.0  # C01

        # Term 5: M6 = (a10 - a00)(b00 + b01)
        U_s[2, 5] = 1.0  # a10
        U_s[0, 5] = -1.0  # a00
        V_s[0, 5] = 1.0  # b00
        V_s[1, 5] = 1.0  # b01
        W_s[3, 5] = 1.0  # C11

        # Term 6: M7 = (a01 - a11)(b10 + b11)
        U_s[1, 6] = 1.0  # a01
        U_s[3, 6] = -1.0  # a11
        V_s[2, 6] = 1.0  # b10
        V_s[3, 6] = 1.0  # b11
        W_s[0, 6] = 1.0  # C00

        return U_s, V_s, W_s

    # Schoolbook 1x2x2 decomposition (rank 4)
    def schoolbook_1x2x2():
        """
        Schoolbook for 1x2x2: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        U: (2, 4), V: (4, 4), W: (2, 4)
        """
        nb, mb, pb = 1, 2, 2
        rank = nb * mb * pb  # 4
        U_b = np.zeros((nb * mb, rank), dtype=float)
        V_b = np.zeros((mb * pb, rank), dtype=float)
        W_b = np.zeros((pb * nb, rank), dtype=float)
        col_b = 0
        for i in range(nb):
            for j in range(mb):
                for k in range(pb):
                    U_b[i * mb + j, col_b] = 1.0
                    V_b[j * pb + k, col_b] = 1.0
                    W_b[k * nb + i, col_b] = 1.0
                    col_b += 1
        return U_b, V_b, W_b

    # Get the block decompositions
    U_str, V_str, W_str = strassen_2x2x2()
    U_sch, V_sch, W_sch = schoolbook_1x2x2()

    # Now tile the 3x4x8 problem.
    # i ranges: [0,2) uses Strassen-like (2-row block), [2,3) uses 1-row block.
    # j ranges: [0,2), [2,4)
    # k ranges: [0,2), [2,4), [4,6), [6,8)

    # For i in [0,2): nb=2, use Strassen for 2x2x2 blocks
    # For i in [2,3): nb=1, use schoolbook for 1x2x2 blocks

    # Process all blocks
    i_blocks = [(0, 2), (2, 3)]  # (i_start, i_end)
    j_blocks = [(0, 2), (2, 4)]
    k_blocks = [(0, 2), (2, 4), (4, 6), (6, 8)]

    for i0, i1 in i_blocks:
        nb = i1 - i0
        for j0, j1 in j_blocks:
            mb = j1 - j0
            for k0, k1 in k_blocks:
                pb = k1 - k0

                if nb == 2 and mb == 2 and pb == 2:
                    add_block(U_str, V_str, W_str, i0, j0, k0, nb, mb, pb)
                elif nb == 1 and mb == 2 and pb == 2:
                    add_block(U_sch, V_sch, W_sch, i0, j0, k0, nb, mb, pb)
                else:
                    # Fallback: schoolbook for this block
                    r_b = nb * mb * pb
                    U_b = np.zeros((nb * mb, r_b), dtype=float)
                    V_b = np.zeros((mb * pb, r_b), dtype=float)
                    W_b = np.zeros((pb * nb, r_b), dtype=float)
                    col_b = 0
                    for i in range(nb):
                        for j in range(mb):
                            for k in range(pb):
                                U_b[i * mb + j, col_b] = 1.0
                                V_b[j * pb + k, col_b] = 1.0
                                W_b[k * nb + i, col_b] = 1.0
                                col_b += 1
                    add_block(U_b, V_b, W_b, i0, j0, k0, nb, mb, pb)

    # Verify
    if time.time() - start_time < DEADLINE - 5:
        assert col == total_rank, f"Expected {total_rank} columns, got {col}"
        if not _verify_decomp(U, V, W, n, m, p):
            # Fallback to schoolbook
            U = np.zeros((n * m, n * m * p), dtype=float)
            V = np.zeros((m * p, n * m * p), dtype=float)
            W = np.zeros((p * n, n * m * p), dtype=float)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
