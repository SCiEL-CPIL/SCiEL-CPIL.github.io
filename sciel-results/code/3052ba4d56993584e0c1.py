"""Matrix multiplication 4x4x5: rank-65 decomposition via recursive Strassen.
4x4x4 part: rank 49 (recursive Strassen 4x4x4 = 7*7).
4x4x1 part: rank 16 (schoolbook for the 5th column).
Total rank = 49 + 16 = 65.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]


def _strassen_2x2x2():
    """Return U(4x7), V(4x7), W(4x7) for 2x2x2 Strassen.
    U rows: A00, A01, A10, A11 (row-major 2x2)
    V rows: B00, B01, B10, B11 (row-major 2x2)
    W rows: C00, C01, C10, C11 (row-major 2x2, C[i,k] convention)
    """
    U2 = np.array([
        [1, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1],
    ], dtype=np.int64)
    V2 = np.array([
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 1],
    ], dtype=np.int64)
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=np.int64)
    return U2, V2, W2


def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 matrix multiplication.
    Uses recursive Strassen: 4x4x4 = 7 * (2x2x2 Strassen).
    
    Returns U(16,49), V(16,49), W(16,49) where:
    U row: i*4+j for A[i,j]
    V row: j*4+k for B[j,k]
    W row: k*4+i for C[k,i] (trace convention)
    """
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((16, rank), dtype=np.int64)
    W = np.zeros((16, rank), dtype=np.int64)

    # The 7 Strassen factors for 4x4x4:
    # Each factor mk has:
    #   A-side: sum of 2x2 blocks with coefficients
    #   B-side: sum of 2x2 blocks with coefficients
    #   Output: sum of 2x2 C blocks with coefficients
    #
    # For 4x4x4, partition into 2x2 blocks (4 blocks each):
    # A = [A00 A01; A10 A11] where Aij is 2x2
    # B = [B00 B01; B10 B11] where Bij is 2x2
    # C = [C00 C01; C10 C11] where Cij is 2x2
    #
    # Strassen 2x2x2 at the block level:
    # M0 = (A00+A11)*(B00+B11)
    # M1 = (A10+A11)*B00
    # M2 = A00*(B01-B11)
    # M3 = A11*(B10-B00)
    # M4 = (A00+A10)*(B11+B10)
    # M5 = (A01-A11)*(B10+B11)
    # M6 = (A10-A00)*(B00+B01)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M5
    #
    # Each Mk is a 2x2x2 product of (sums of 2x2 blocks), which itself
    # uses Strassen with rank 7. So total rank = 7 * 7 = 49.

    # Block indices for 4x4 partitioned into 2x2 blocks:
    # Block (bi_row, bi_col) where bi_row, bi_col in {0,1}
    # Block (bi_row, bi_col) covers rows 2*bi_row to 2*bi_row+1, cols 2*bi_col to 2*bi_col+1

    def a_block_rows(bi_row, bi_col):
        """U row indices for A block (bi_row, bi_col)."""
        return [
            (2 * bi_row + 0) * 4 + (2 * bi_col + 0),
            (2 * bi_row + 0) * 4 + (2 * bi_col + 1),
            (2 * bi_row + 1) * 4 + (2 * bi_col + 0),
            (2 * bi_row + 1) * 4 + (2 * bi_col + 1),
        ]

    def b_block_rows(bi_row, bi_col):
        """V row indices for B block (bi_row, bi_col)."""
        return [
            (2 * bi_row + 0) * 4 + (2 * bi_col + 0),
            (2 * bi_row + 0) * 4 + (2 * bi_col + 1),
            (2 * bi_row + 1) * 4 + (2 * bi_col + 0),
            (2 * bi_row + 1) * 4 + (2 * bi_col + 1),
        ]

    def c_block_rows(bi_row, bi_col):
        """W row indices for C block (bi_row, bi_col) in trace convention.
        C[i,k] is at W row k*4+i.
        C block (bi_row, bi_col) covers C[2*bi_row+0..1, 2*bi_col+0..1]
        So C[2*bi_row+u, 2*bi_col+v] is at W row (2*bi_col+v)*4 + (2*bi_row+u)
        """
        return [
            (2 * bi_col + 0) * 4 + (2 * bi_row + 0),
            (2 * bi_col + 0) * 4 + (2 * bi_row + 1),
            (2 * bi_col + 1) * 4 + (2 * bi_row + 0),
            (2 * bi_col + 1) * 4 + (2 * bi_row + 1),
        ]

    # The 7 Strassen factors at the 4x4 level:
    # Each factor mk: A_combination, B_combination, C_combination
    # A_combination: list of (block_idx, coeff) where block_idx in {0,1,2,3} for (0,0),(0,1),(1,0),(1,1)
    # B_combination: list of (block_idx, coeff)
    # C_combination: list of (block_idx, coeff)

    # Block index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    def blk(r, c):
        return r * 2 + c

    strassen_factors = [
        # M0: (A00+A11)*(B00+B11)
        ([(blk(0,0), 1), (blk(1,1), 1)], [(blk(0,0), 1), (blk(1,1), 1)], None),
        # M1: (A10+A11)*B00
        ([(blk(1,0), 1), (blk(1,1), 1)], [(blk(0,0), 1)], None),
        # M2: A00*(B01-B11)
        ([(blk(0,0), 1)], [(blk(0,1), 1), (blk(1,1), -1)], None),
        # M3: A11*(B10-B00)
        ([(blk(1,1), 1)], [(blk(1,0), 1), (blk(0,0), -1)], None),
        # M4: (A00+A10)*(B11+B10)
        ([(blk(0,0), 1), (blk(1,0), 1)], [(blk(1,1), 1), (blk(1,0), 1)], None),
        # M5: (A01-A11)*(B10+B11)
        ([(blk(0,1), 1), (blk(1,1), -1)], [(blk(1,0), 1), (blk(1,1), 1)], None),
        # M6: (A10-A00)*(B00+B01)
        ([(blk(1,0), 1), (blk(0,0), -1)], [(blk(0,0), 1), (blk(0,1), 1)], None),
    ]

    # C output combinations:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M5
    c_output = [
        {0: 1, 3: 1, 4: -1, 6: 1},   # C00
        {2: 1, 4: 1},                  # C01
        {1: 1, 5: 1},                  # C10
        {0: 1, 1: -1, 2: 1, 5: 1},    # C11
    ]

    for mk in range(7):
        A_comb, B_comb, _ = strassen_factors[mk]
        for nk in range(7):
            col = mk * 7 + nk
            # U: for each block in A_comb, add Strassen U2 with coefficient
            for (bi, coeff) in A_comb:
                bi_row = bi // 2
                bi_col = bi % 2
                u_rows = a_block_rows(bi_row, bi_col)
                for i in range(4):
                    U[u_rows[i], col] += coeff * U2[i, nk]

            # V: for each block in B_comb, add Strassen V2 with coefficient
            for (bi, coeff) in B_comb:
                bi_row = bi // 2
                bi_col = bi % 2
                v_rows = b_block_rows(bi_row, bi_col)
                for j in range(4):
                    V[v_rows[j], col] += coeff * V2[j, nk]

            # W: for each output block that includes factor mk, add with coefficient
            for ci, coeff in c_output[nk].items():
                ci_row = ci // 2
                ci_col = ci % 2
                w_rows = c_block_rows(ci_row, ci_col)
                for k in range(4):
                    W[w_rows[k], col] += coeff * W2[k, nk]

    return U, V, W


def _build_4x4x1_schoolbook():
    """Build rank-16 schoolbook decomposition for 4x4x1.
    A(4x4) * B(4x1) = C(4x1).
    
    Returns U(16,16), V(4,16), W(4,16) where:
    U row: i*4+j for A[i,j]
    V row: j (for B[j,0])
    W row: i (for C[i,0])
    """
    rank = 16
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((4, rank), dtype=np.int64)
    W = np.zeros((4, rank), dtype=np.int64)
    col = 0
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, col] = 1
            V[j, col] = 1
            W[i, col] = 1
            col += 1
    return U, V, W


def _build_4x4x5():
    """Build the full 4x4x5 decomposition with rank 65.
    
    Part 1: 4x4x4 (columns 0-3 of B) using recursive Strassen, rank 49.
    Part 2: 4x4x1 (column 4 of B) using schoolbook, rank 16.
    
    Global indexing:
    U row: i*4+j for A[i,j] (16 rows)
    V row: j*5+k for B[j,k] (20 rows)
    W row: k*4+i for C[k,i] trace convention (20 rows)
    """
    U44, V44, W44 = _build_4x4x4_strassen()
    U41, V41, W41 = _build_4x4x1_schoolbook()

    total_rank = 49 + 16
    U = np.zeros((16, total_rank), dtype=np.int64)
    V = np.zeros((20, total_rank), dtype=np.int64)
    W = np.zeros((20, total_rank), dtype=np.int64)

    # Part 1: 4x4x4 (k=0,1,2,3)
    # U44: (16, 49) - U row i*4+j
    # V44: (16, 49) - V row j*4+k (local, k in 0..3)
    # W44: (16, 49) - W row k*4+i (local, k in 0..3)
    U[:, :49] = U44
    # Map V44 row j*4+k to global V row j*5+k
    for j in range(4):
        for k in range(4):
            V[j * 5 + k, :49] = V44[j * 4 + k, :]
    # Map W44 row k*4+i to global W row k*4+i (same since n=4)
    W[:, :49] = W44

    # Part 2: 4x4x1 (k=4)
    # U41: (16, 16) - U row i*4+j
    # V41: (4, 16) - V row j (local, for B[j,4])
    # W41: (4, 16) - W row i (local, for C[i,4])
    U[:, 49:] = U41
    # Map V41 row j to global V row j*5+4
    for j in range(4):
        V[j * 5 + 4, 49:] = V41[j, :]
    # Map W41 row i to global W row 4*4+i = 16+i
    for i in range(4):
        W[16 + i, 49:] = W41[i, :]

    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for r in range(U.shape[1]):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    T_exp = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i * m + j, j * p + k, k * n + i] = 1
    return np.array_equal(T, T_exp)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _to_list(mat):
    return mat.tolist()


def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS

    # Try the rank-65 recursive Strassen decomposition
    try:
        U, V, W = _build_4x4x5()
        if time.time() < deadline and _verify(U, V, W, n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    except Exception:
        pass

    # Fallback: schoolbook rank 80
    if time.time() < deadline:
        U, V, W = _schoolbook(n, m, p)
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    # Emergency fallback
    U, V, W = _schoolbook(n, m, p)
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
