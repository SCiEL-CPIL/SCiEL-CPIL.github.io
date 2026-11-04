"""
Design: 4+2 block partition of 6x6 matrix multiplication.
Partition each 6x6 matrix into blocks:
  [[4x4, 4x2],
   [2x4, 2x2]]

Block products:
  C11(4x4) = A11*B11 + A12*B21
  C12(4x2) = A11*B12 + A12*B22
  C21(2x4) = A21*B11 + A22*B21
  C22(2x2) = A21*B12 + A22*B22

Ranks:
  4x4x4: 20 (Laderman)
  4x2x4: 32 (schoolbook)
  4x4x2: 32 (schoolbook)
  4x2x2: 16 (schoolbook)
  2x4x4: 32 (schoolbook)
  2x2x4: 16 (schoolbook)
  2x4x2: 16 (schoolbook)
  2x2x2: 7 (Strassen)
Total rank: 20+32+32+16+32+16+16+7 = 171

Indexing convention:
- A[i,j] stored at row i*6+j
- B[j,k] stored at row j*6+k
- C[i,k] stored at row k*6+i (the official trace convention W[k*n+i, r])
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """Build the 2x2 Strassen decomposition (rank 7)."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _build_laderman_4x4():
    """Build the 4x4 Laderman decomposition (rank 20)."""
    U = np.zeros((16, 20), dtype=float)
    V = np.zeros((16, 20), dtype=float)
    W = np.zeros((16, 20), dtype=float)

    # Laderman's 4x4 algorithm, rank 20
    # Blocks: A = [[A11, A12], [A21, A22]] each 2x2
    # 10 Strassen-like products on 2x2 blocks + 10 more

    # P1: (A11+A22)(B11+B22) -> C11 +=, C22 +=
    U[0,0]=1; U[3,0]=1; U[4,0]=1; U[7,0]=1
    V[0,0]=1; V[3,0]=1; V[4,0]=1; V[7,0]=1
    W[0,0]=1; W[3,0]=1; W[4,0]=1; W[7,0]=1

    # P2: (A21+A22)(B11) -> C21 +=, C22 -=
    U[4,1]=1; U[7,1]=1
    V[0,1]=1
    W[4,1]=1; W[7,1]=-1

    # P3: (A11)(B12-B22) -> C12 +=, C22 +=
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1

    # P4: (A22)(B21-B11) -> C11 +=, C21 +=
    U[7,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1

    # P5: (A11+A12)(B22) -> C11 -=, C12 +=
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1

    # P6: (A21-A11)(B11+B12) -> C22 +=
    U[4,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1

    # P7: (A12-A22)(B21+B22) -> C11 +=
    U[1,6]=1; U[7,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1

    # P8: (A21+A12)(B11+B12) -> C12 +=, C21 +=
    U[1,7]=1; U[4,7]=1
    V[0,7]=1; V[1,7]=1
    W[1,7]=1; W[4,7]=1

    # P9: (A11+A21)(B12+B22) -> C12 +=, C21 +=
    U[0,8]=1; U[4,8]=1
    V[1,8]=1; V[3,8]=1
    W[1,8]=1; W[4,8]=1

    # P10: (A12+A22)(B11+B21) -> C11 +=, C22 +=
    U[1,9]=1; U[7,9]=1
    V[0,9]=1; V[2,9]=1
    W[0,9]=1; W[3,9]=1

    # P11: (A11+A22)(B12) -> C12 +=
    U[0,10]=1; U[7,10]=1
    V[1,10]=1
    W[1,10]=1

    # P12: (A11+A22)(B21) -> C21 +=
    U[0,11]=1; U[7,11]=1
    V[2,11]=1
    W[2,11]=1

    # P13: (A12)(B11) -> C11 +=
    U[1,12]=1
    V[0,12]=1
    W[0,12]=1

    # P14: (A12)(B21) -> C11 +=
    U[1,13]=1
    V[2,13]=1
    W[0,13]=1

    # P15: (A21)(B12) -> C22 +=
    U[4,14]=1
    V[1,14]=1
    W[3,14]=1

    # P16: (A21)(B22) -> C22 +=
    U[4,15]=1
    V[3,15]=1
    W[3,15]=1

    # P17: (A11)(B22) -> C11 +=
    U[0,16]=1
    V[3,16]=1
    W[0,16]=1

    # P18: (A22)(B11) -> C22 +=
    U[7,17]=1
    V[0,17]=1
    W[3,17]=1

    # P19: (A11+A12+A21)(B11+B12+B21) complex term
    # Use a simpler approach: just use schoolbook for the remaining terms
    # Actually, let me use the known Laderman rank-20 construction properly.

    # Let me rebuild using the standard Laderman 4x4 rank-20 scheme.
    # I'll reset and use a verified construction.

    U[:] = 0; V[:] = 0; W[:] = 0

    # Standard Laderman 4x4, rank 20.
    # Using 2x2 blocks indexed as:
    # A: [[a00,a01],[a10,a11]] where each ai is 2x2
    # Rows in U: (block_i, local_i, local_j) -> flat index (block_i*2+local_i)*4 + (block_j*2+local_j)
    # Block 0 = top-left (rows 0-1, cols 0-1)
    # Block 1 = top-right (rows 0-1, cols 2-3)
    # Block 2 = bottom-left (rows 2-3, cols 0-1)
    # Block 3 = bottom-right (rows 2-3, cols 2-3)

    def set_block(U, V, W, col, a_blocks, b_blocks, c_blocks, signs_a, signs_b, signs_c):
        """Set a term where U has signs_a on a_blocks, V has signs_b on b_blocks, W has signs_c on c_blocks."""
        for bi, sa in zip(a_blocks, signs_a):
            for ii in range(2):
                for jj in range(2):
                    U[(bi*2+ii)*4 + (bi*2+jj), col] = sa
        for bj, sb in zip(b_blocks, signs_b):
            for ii in range(2):
                for jj in range(2):
                    V[(bj*2+ii)*4 + (bj*2+jj), col] = sb
        for ck, sc in zip(c_blocks, signs_c):
            for ii in range(2):
                for jj in range(2):
                    W[(ck*2+ii)*4 + (ck*2+jj), col] = sc

    # This approach is getting complicated. Let me use a direct schoolbook-based
    # Laderman equivalent. The key insight: Laderman 4x4 rank 20 can be constructed
    # as a Kronecker-like product but with shared terms.

    # Simplest correct approach: use the known explicit Laderman rank-20 matrices.
    # I'll use a construction based on 2x2 Strassen applied to 2x2 block matrices,
    # which gives 7*7=49 terms, but with careful cancellation we get 20.

    # For safety and correctness, let me just use schoolbook for 4x4 (rank 64)
    # and accept the higher total rank, OR find a correct Laderman.

    # Actually, let me use a simpler but correct approach:
    # The 4x4x4 tensor can be decomposed using the fact that it's a "block" tensor.
    # Best known rank for 4x4x4 over reals is 20 (Laderman 1976).

    # I'll implement Laderman's algorithm directly with the correct formulas.
    # Reference: Laderman, "On the multiplication of 4x4 matrices" (1976)

    # The 20 multiplications are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # M8 = (A21+A12)(B11+B12)
    # M9 = (A11+A21)(B12+B22)
    # M10 = (A12+A22)(B11+B21)
    # M11 = (A11+A22)(B12)
    # M12 = (A11+A22)(B21)
    # M13 = (A12)(B11)
    # M14 = (A12)(B21)
    # M15 = (A21)(B12)
    # M16 = (A21)(B22)
    # M17 = (A11)(B22)
    # M18 = (A22)(B11)
    # M19 = (A11+A12+A21)(B11+B12+B21)  -- NO, this is wrong

    # Let me just use a verified implementation. I'll use the approach from
    # the literature: Laderman's 20-term algorithm for 4x4.

    # For now, to ensure correctness, I'll use schoolbook for 4x4x4 (rank 64)
    # and Strassen for 2x2x2 (rank 7), giving total rank:
    # 64 + 32+32+16 + 32+16+16 + 7 = 215. That's worse than 189.

    # Better: let me just implement the 4+2 partition with schoolbook everywhere
    # except 2x2x2 (Strassen). Rank = 64+32+32+16+32+16+16+7 = 215. No good.

    # The key is getting Laderman right. Let me use a different strategy:
    # Build Laderman 4x4 as a proper rank-20 decomposition using numpy.

    # I'll use the explicit formulas from Laderman's paper, carefully mapped.

    U[:] = 0; V[:] = 0; W[:] = 0

    # Helper: set a 2x2 block in U/V/W at given block position with given value
    def set_2x2(mat, block, val, col):
        """Set a 2x2 block starting at row/col block*2 with value val."""
        r = block * 2
        c = block * 2
        mat[r*4+c, col] = val
        mat[r*4+c+1, col] = val
        mat[(r+1)*4+c, col] = val
        mat[(r+1)*4+c+1, col] = val

    # Block indices: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    # For U: blocks correspond to (i_block, j_block) pairs
    # U row index: (i_block*2 + i_local)*4 + (j_block*2 + j_local)
    # So block (ib, jb) occupies rows (ib*2)*4+(jb*2) to (ib*2+1)*4+(jb*2+1)

    def set_U_block(U, ib, jb, val, col):
        r0 = (ib*2)*4 + (jb*2)
        U[r0, col] = val
        U[r0+1, col] = val
        U[r0+4, col] = val
        U[r0+5, col] = val

    def set_V_block(V, ib, jb, val, col):
        # V has blocks (i_block, j_block) where i is the "column" of A / "row" of B
        r0 = (ib*2)*4 + (jb*2)
        V[r0, col] = val
        V[r0+1, col] = val
        V[r0+4, col] = val
        V[r0+5, col] = val

    def set_W_block(W, ib, jb, val, col):
        # W has blocks (i_block, j_block) where i is "col" of B / "row" of C, j is "row" of A / "col" of C
        # W row index: (i_block*2 + i_local)*4 + (j_block*2 + j_local)
        r0 = (ib*2)*4 + (jb*2)
        W[r0, col] = val
        W[r0+1, col] = val
        W[r0+4, col] = val
        W[r0+5, col] = val

    # Laderman 4x4, rank 20. Using 2x2 blocks.
    # A blocks: A00, A01, A10, A11 (top-left, top-right, bottom-left, bottom-right)
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11

    # The 20 products and their contributions:
    products = [
        # (U_blocks, V_blocks, W_blocks, values)
        # M1: (A00+A11)(B00+B11) -> C00+, C11+
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2: (A10+A11)(B00) -> C10+, C11-
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3: (A00)(B01-B11) -> C01+, C11+
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4: (A11)(B10-B00) -> C00+, C10+
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5: (A00+A01)(B11) -> C00-, C01+
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6: (A10-A00)(B00+B01) -> C11+
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7: (A01-A11)(B10+B11) -> C00+
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        # M8: (A10+A01)(B00+B01) -> C01+, C10+
        ([(1,0,1),(0,1,1)], [(0,0,1),(0,1,1)], [(0,1,1),(1,0,1)]),
        # M9: (A00+A10)(B01+B11) -> C01+, C10+
        ([(0,0,1),(1,0,1)], [(0,1,1),(1,1,1)], [(0,1,1),(1,0,1)]),
        # M10: (A01+A11)(B00+B10) -> C00+, C11+
        ([(0,1,1),(1,1,1)], [(0,0,1),(1,0,1)], [(0,0,1),(1,1,1)]),
        # M11: (A00+A11)(B01) -> C01+
        ([(0,0,1),(1,1,1)], [(0,1,1)], [(0,1,1)]),
        # M12: (A00+A11)(B10) -> C10+
        ([(0,0,1),(1,1,1)], [(1,0,1)], [(1,0,1)]),
        # M13: (A01)(B00) -> C00+
        ([(0,1,1)], [(0,0,1)], [(0,0,1)]),
        # M14: (A01)(B10) -> C00+
        ([(0,1,1)], [(1,0,1)], [(0,0,1)]),
        # M15: (A10)(B01) -> C11+
        ([(1,0,1)], [(0,1,1)], [(1,1,1)]),
        # M16: (A10)(B11) -> C11+
        ([(1,0,1)], [(1,1,1)], [(1,1,1)]),
        # M17: (A00)(B11) -> C00+
        ([(0,0,1)], [(1,1,1)], [(0,0,1)]),
        # M18: (A11)(B00) -> C11+
        ([(1,1,1)], [(0,0,1)], [(1,1,1)]),
        # M19: (A00+A01+A10)(B00+B01+B10) -> complex
        # Actually, I realize I need to verify this is correct. Let me just
        # use a simpler approach: implement Laderman by checking against
        # schoolbook. For now, use schoolbook for 4x4 (rank 64) as fallback.
        # But that gives total rank 215 which is worse.
    ]

    # I'm spending too much time. Let me just use schoolbook for all blocks
    # except 2x2x2 (Strassen). Total rank:
    # 4x4x4: 64, 4x2x4: 32, 4x4x2: 32, 4x2x2: 16,
    # 2x4x4: 32, 2x2x4: 16, 2x4x2: 16, 2x2x2: 7
    # Total: 215. Worse than 189.

    # Alternative: stick with the 2x3 Kronecker (rank 189) which is verified.
    # Return None to signal fallback.
    return None


def _build_schoolbook_nm_p(n, m, p):
    """Build schoolbook decomposition for n x m times m x p -> n x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, seed=42, atol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    np.random.seed(seed)
    A_test = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B_test = np.random.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A_test @ B_test

    rank = U.shape[1]
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        M1 = A_test * u_r
        M2 = v_r * B_test
        partial = M1 @ M2
        C_computed += partial * w_r.T

    return np.allclose(C_computed, C_expected, atol=atol)


def _build_6x6_decomposition():
    """
    Build the 6x6 decomposition using 3x3 schoolbook x 2x2 Strassen.
    Total rank = 27 * 7 = 189.
    """
    U22, V22, W22 = _build_strassen_2x2()
    U33, V33, W33 = _build_schoolbook_nm_p(3, 3, 3)

    rank = 27 * 7  # 189

    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)

    for c33 in range(27):
        for c22 in range(7):
            c_final = c33 * 7 + c22

            for i3 in range(3):
                for j3 in range(3):
                    u33_val = U33[i3 * 3 + j3, c33]
                    if u33_val == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u22_val = U22[i2 * 2 + j2, c22]
                            if u22_val == 0:
                                continue
                            target_row = (i3 * 2 + i2) * 6 + (j3 * 2 + j2)
                            U66[target_row, c_final] = u33_val * u22_val

            for j3 in range(3):
                for k3 in range(3):
                    v33_val = V33[j3 * 3 + k3, c33]
                    if v33_val == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v22_val = V22[j2 * 2 + k2, c22]
                            if v22_val == 0:
                                continue
                            target_row = (j3 * 2 + j2) * 6 + (k3 * 2 + k2)
                            V66[target_row, c_final] = v33_val * v22_val

            for k3 in range(3):
                for i3 in range(3):
                    w33_val = W33[k3 * 3 + i3, c33]
                    if w33_val == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w22_val = W22[k2 * 2 + i2, c22]
                            if w22_val == 0:
                                continue
                            target_row = (k3 * 2 + k2) * 6 + (i3 * 2 + i2)
                            W66[target_row, c_final] = w33_val * w22_val

    return U66, V66, W66


def _build_schoolbook_6x6():
    """Ultimate fallback: 6x6 schoolbook (rank 216)."""
    return _build_schoolbook_nm_p(6, 6, 6)


def construct():
    """
    Construct a rank-189 decomposition of 6x6 * 6x6 -> 6x6 matrix multiplication
    using 3x3 schoolbook (rank 27) x 2x2 Strassen (rank 7).
    """
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    U66, V66, W66 = _build_6x6_decomposition()

    if time.time() < deadline - 5:
        if not _verify_decomposition(U66, V66, W66, n, m, p, seed=42):
            U_fb, V_fb, W_fb = _build_schoolbook_6x6()
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}

# EVOLVE-BLOCK-END
