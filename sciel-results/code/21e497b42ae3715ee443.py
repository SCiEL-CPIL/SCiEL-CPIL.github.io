"""Combined program: split-column strategy.
4x4x5 = 4x4x4 (rank-49 block Strassen) + 4x4x1 (rank-16 schoolbook).
Verified with NumPy; falls back to schoolbook rank 80 if Strassen is invalid."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return U (4x7), V (4x7), W (4x7) for 2x2x2 Strassen.
    U rows: A00, A01, A10, A11 ; V rows: B00, B01, B10, B11 ; W rows: C00, C01, C10, C11."""
    U = np.array([
        [1, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1],
    ], dtype=np.int64)
    V = np.array([
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 1],
    ], dtype=np.int64)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=np.int64)
    return U, V, W


def _build_4x4x4_strassen():
    """Build 4x4x4 block-Strassen decomposition (rank 49).
    U: (16,49) row i*4+j = A[i,j]
    V: (16,49) row j*4+k = B[j,k]
    W: (16,49) row k*4+i = C[k,i]  (trace convention W[k*n+i, r], n=4)
    Returns (U,V,W) or None on construction error.
    """
    n = 4
    U, V, W = (np.zeros((n*n, 49), dtype=np.int64) for _ in range(3))
    u2, v2, w2 = _strassen_2x2()

    def add_subproblem(a_rows, b_rows, c_rows, col_offset):
        for col in range(7):
            for i in range(4):
                U[a_rows[i], col_offset + col] += u2[i, col]
                V[b_rows[i], col_offset + col] += v2[i, col]
                W[c_rows[i], col_offset + col] += w2[i, col]

    # 2x2 block element positions in the 4x4 (row-major) indexing.
    # A[i,j] -> i*4+j ; B[j,k] -> j*4+k ; C(k,i) -> k*4+i
    def A(i, j): return i * 4 + j
    def B(j, k): return j * 4 + k
    def C(k, i): return k * 4 + i

    # The 7 Strassen multiplications. Each is a 2x2x2 sub-problem.
    # a_rows = [A_sub00, A_sub01, A_sub10, A_sub11]  (each a U row index)
    # b_rows = [B_sub00, B_sub01, B_sub10, B_sub11]  (each a V row index)
    # c_rows = [C_sub00, C_sub01, C_sub10, C_sub11]  (each a W row index)
    # For sums/differences we add two sub-problems with +/- signs.

    # M1 = (A00+A11)*(B00+B11) -> contributes to C00 and C11
    # M2 = (A10+A11)*B00       -> contributes to C10 and C11
    # M3 = A00*(B01-B11)       -> contributes to C01 and C11
    # M4 = A11*(B10-B00)       -> contributes to C00 and C10
    # M5 = (A00+A01)*B11       -> contributes to C00 and C01
    # M6 = (A10-A00)*(B00+B01) -> contributes to C11
    # M7 = (A01-A11)*(B10+B11) -> contributes to C00

    # We express each Mi as a sum of 2x2x2 tensor products.
    # Because Strassen U/V/W are linear, we can write each Mi as a
    # sum over elementary (single A element)*(single B element) 2x2x2 problems.

    def add_term(a_idx, b_idx, c_idx, col_offset, sign=1):
        """Add a single elementary 2x2x2 problem: A element a_idx, B element b_idx,
        producing C element c_idx, using the 2x2 Strassen pattern scaled by sign."""
        # This is NOT a full 2x2x2 Strassen; it's one scalar multiplication.
        # For a scalar mult a*b -> c, the rank-1 tensor is U[a_idx]=1, V[b_idx]=1, W[c_idx]=1.
        # But we want to reuse the 2x2 Strassen structure. Instead, build the 49-term
        # decomposition directly via the block formulas below.
        pass

    # Direct construction: enumerate the 49 rank-1 terms of block Strassen.
    # Block Strassen for 4x4x4: 7 multiplications, each a 2x2x2 Strassen (rank 7).
    # Total 49 rank-1 terms. Each term t in multiplication m and Strassen slot s:
    #   U_row = the A element feeding slot s of multiplication m
    #   V_row = the B element feeding slot s of multiplication m
    #   W_row = the C element that slot s of multiplication m contributes to (with sign)
    #
    # We build this by, for each of the 7 multiplications, applying the 2x2 Strassen
    # to the (possibly summed) 2x2 blocks. Since U,V,W are linear, a summed block
    # (e.g. A00+A11) means the same Strassen slot gets contributions from both A00 and A11.
    #
    # Concretely, for multiplication m with A-blocks a = [a00,a01,a10,a11] (each a list of
    # U row indices with signs) and B-blocks b = [b00,b01,b10,b11], and C-blocks
    # c = [c00,c01,c10,c11] (each a list of W row indices with signs):
    # for each Strassen slot s (0..6) and each (ai in a[s_a], bj in b[s_b], cj in c[s_c]):
    #     U[ai, col+s] += sign_ai * u2[s_a, s]
    #     V[bj, col+s] += sign_bj * v2[s_b, s]
    #     W[cj, col+s] += sign_cj * w2[s_c, s]
    # where s_a, s_b, s_c are the 2x2 block positions used by Strassen slot s.

    # Strassen slot -> (A position, B position, C position)
    # Standard Strassen:
    # s0: M0 = A00*B00        (a=00,b=00,c=00)
    # s1: M1 = A10*B00        (a=10,b=00,c=10)
    # s2: M2 = A00*B01        (a=00,b=01,c=01)
    # s3: M3 = A10*B11        (a=10,b=11,c=11)
    # s4: M4 = (A00+A11)*(B00+B11)  (a=00&11,b=00&11,c=00&11)
    # s5: M5 = (A10-A00)*(B00+B01)  (a=10&00,b=00&01,c=10&11)
    # s6: M6 = (A01-A11)*(B10+B11)  (a=01&11,b=10&11,c=01&10)
    # This is getting complicated. Let me use the cleaner standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M5 + M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2x2 multiplication. I'll build each Mi by applying the 2x2 Strassen
    # to its 2x2 A-block and 2x2 B-block, then mapping the 2x2 C-block output to the
    # appropriate 4x4 C-block with the correct sign.

    # For each Mi, define:
    #   a_block: dict (pos)->list of (u_row, sign) for the 2x2 A positions
    #   b_block: dict (pos)->list of (v_row, sign)
    #   c_block: dict (pos)->list of (w_row, sign)
    # Then for each Strassen slot, combine.

    # 2x2 Strassen slots (using u2,v2,w2 above):
    # slot s uses A-positions per column s of u2, B-positions per column s of v2,
    # C-positions per column s of w2. Each row of u2/v2/w2 is a 2x2 position (00,01,10,11).
    # So for slot s: A-positions = rows i where u2[i,s]!=0, B-positions = rows j where v2[j,s]!=0,
    # C-positions = rows k where w2[k,s]!=0.

    def apply_mult(m_index, a_block, b_block, c_block, col_offset):
        """a_block, b_block, c_block: dict from 2x2 position (0..3) to list of (row, sign)."""
        for s in range(7):
            for i in range(4):  # A 2x2 position
                if u2[i, s] != 0:
                    for (arow, asgn) in a_block.get(i, []):
                        U[arow, col_offset + s] += asgn * u2[i, s]
            for j in range(4):  # B 2x2 position
                if v2[j, s] != 0:
                    for (brow, bsgn) in b_block.get(j, []):
                        V[brow, col_offset + s] += bsgn * v2[j, s]
            for k in range(4):  # C 2x2 position
                if w2[k, s] != 0:
                    for (crow, csgn) in c_block.get(k, []):
                        W[crow, col_offset + s] += csgn * w2[k, s]

    # 2x2 positions: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # A00 = A[0:2,0:2]: pos0=A(0,0), pos1=A(0,1), pos2=A(1,0), pos3=A(1,1)
    # A01 = A[0:2,2:4]: pos0=A(0,2), pos1=A(0,3), pos2=A(1,2), pos3=A(1,3)
    # A10 = A[2:4,0:2]: pos0=A(2,0), pos1=A(2,1), pos2=A(3,0), pos3=A(3,1)
    # A11 = A[2:4,2:4]: pos0=A(2,2), pos1=A(2,3), pos2=A(3,2), pos3=A(3,3)
    def a_block_of(block):
        # block in {00,01,10,11}
        r0, c0 = (0, 0) if block == '00' else (0, 2) if block == '01' else (2, 0) if block == '10' else (2, 2)
        return {
            0: [(A(r0, c0), 1)],
            1: [(A(r0, c0 + 1), 1)],
            2: [(A(r0 + 1, c0), 1)],
            3: [(A(r0 + 1, c0 + 1), 1)],
        }

    def b_block_of(block):
        r0, c0 = (0, 0) if block == '00' else (0, 2) if block == '01' else (2, 0) if block == '10' else (2, 2)
        return {
            0: [(B(r0, c0), 1)],
            1: [(B(r0, c0 + 1), 1)],
            2: [(B(r0 + 1, c0), 1)],
            3: [(B(r0 + 1, c0 + 1), 1)],
        }

    # C-blocks: C(k,i) = k*4+i. C00 = C[0:2,0:2]: pos0=C(0,0),pos1=C(0,1),pos2=C(1,0),pos3=C(1,1)
    # C01 = C[0:2,2:4]: pos0=C(0,2),pos1=C(0,3),pos2=C(1,2),pos3=C(1,3)
    # C10 = C[2:4,0:2]: pos0=C(2,0),pos1=C(2,1),pos2=C(3,0),pos3=C(3,1)
    # C11 = C[2:4,2:4]: pos0=C(2,2),pos1=C(2,3),pos2=C(3,2),pos3=C(3,3)
    def c_block_of(block, sign=1):
        r0, c0 = (0, 0) if block == '00' else (0, 2) if block == '01' else (2, 0) if block == '10' else (2, 2)
        return {
            0: [(C(r0, c0), sign)],
            1: [(C(r0, c0 + 1), sign)],
            2: [(C(r0 + 1, c0), sign)],
            3: [(C(r0 + 1, c0 + 1), sign)],
        }

    # Now define the 7 multiplications.
    # M1 = (A00+A11)*(B00+B11), contributes to C00 (+1) and C11 (+1)
    apply_mult(1,
               {**a_block_of('00'), **a_block_of('11')},  # A00+A11
               {**b_block_of('00'), **b_block_of('11')},  # B00+B11
               {**c_block_of('00', 1), **c_block_of('11', 1)},
               0)
    # M2 = (A10+A11)*B00, contributes to C10 (+1) and C11 (-1)
    apply_mult(2,
               {**a_block_of('10'), **a_block_of('11')},  # A10+A11
               b_block_of('00'),
               {**c_block_of('10', 1), **c_block_of('11', -1)},
               7)
    # M3 = A00*(B01-B11), contributes to C01 (+1) and C11 (+1)
    apply_mult(3,
               a_block_of('00'),
               {**b_block_of('01'), **{k: [(r, -s)] for k, lst in b_block_of('11').items() for (r, s) in lst}},
               {**c_block_of('01', 1), **c_block_of('11', 1)},
               14)
    # M4 = A11*(B10-B00), contributes to C00 (+1) and C10 (+1)
    apply_mult(4,
               a_block_of('11'),
               {**b_block_of('10'), **{k: [(r, -s)] for k, lst in b_block_of('00').items() for (r, s) in lst}},
               {**c_block_of('00', 1), **c_block_of('10', 1)},
               21)
    # M5 = (A00+A01)*B11, contributes to C00 (-1) and C01 (+1)
    apply_mult(5,
               {**a_block_of('00'), **a_block_of('01')},
               b_block_of('11'),
               {**c_block_of('00', -1), **c_block_of('01', 1)},
               28)
    # M6 = (A10-A00)*(B00+B01), contributes to C11 (+1)
    apply_mult(6,
               {**a_block_of('10'), **{k: [(r, -s)] for k, lst in a_block_of('00').items() for (r, s) in lst}},
               {**b_block_of('00'), **b_block_of('01')},
               c_block_of('11', 1),
               35)
    # M7 = (A01-A11)*(B10+B11), contributes to C00 (+1)
    apply_mult(7,
               {**a_block_of('01'), **{k: [(r, -s)] for k, lst in a_block_of('11').items() for (r, s) in lst}},
               {**b_block_of('10'), **b_block_of('11')},
               c_block_of('00', 1),
               42)

    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check that sum_r U[:,r] * V[:,r] * W[:,r] == multiplication tensor."""
    # Build the multiplication tensor T[i,j,k] = 1 if A[i,j]*B[j,k] contributes to C[i,k]
    # In our convention: U row i*m+j, V row j*p+k, W row k*n+i
    # Tensor T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i should equal 1 when
    # the indices are consistent (same j).
    # Reconstruct: for each rank term r, outer product U[:,r] * V[:,r] * W[:,r].
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for r in range(U.shape[1]):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    # Expected: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k; else 0.
    T_exp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1
    return np.array_equal(T, T_exp)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W


def _to_list(mat):
    return mat.tolist()


def construct():
    deadline = time.time() + 120
    n, m, p = DIMENSIONS
    # Split: 4x4x4 (cols 0..3) + 4x4x1 (col 4)
    # Build 4x4x4 Strassen (rank 49) for the first 4 columns of B.
    try:
        U44, V44, W44 = _build_4x4x4_strassen()
    except Exception:
        U44 = V44 = W44 = None

    if U44 is not None and _verify(U44, V44, W44, n, m, 4):
        # Strassen valid for 4x4x4. Append schoolbook for 4x4x1 (col 4 of B).
        U1, V1, W1 = _schoolbook(n, m, 1)
        rank = 49 + 16
        U = np.hstack([U44, U1])
        V = np.hstack([V44, V1])
        W = np.hstack([W44, W1])
        # Verify full 4x4x5.
        if _verify(U, V, W, n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    # Fallback: schoolbook rank 80.
    U, V, W = _schoolbook(n, m, p)
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
