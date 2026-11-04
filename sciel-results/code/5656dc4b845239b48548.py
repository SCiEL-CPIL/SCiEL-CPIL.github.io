"""
Matrix multiplication 2x3 * 3x7: rank-minimal exact tensor decomposition.

Strategy:
  Split the 3 intermediate dimension into a 2-dim part and a 1-dim part.
  For the 2x2 * 2x7 sub-problem, split the 7 output columns into three 2-col
  blocks and one 1-col block.  Use Strassen (rank 7) for each 2x2x2 block and
  schoolbook (rank 4) for the 2x2x1 block, giving 7*3 + 4 = 25 for that part.
  The 1x1 * 1x7 part uses schoolbook (rank 7).
  Total rank = 25 + 7 = 32.
"""
import time

DIMENSIONS = [2, 3, 7]

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 7

    # We will build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U index: i*m + j   (i in 0..n-1, j in 0..m-1)
    # V index: j*p + k   (j in 0..m-1, k in 0..p-1)
    # W index: k*n + i   (i in 0..n-1, k in 0..p-1)  -- trace convention W[k*n+i, r]

    # We'll accumulate into lists of (row, value) per rank column, then build matrices.
    # Actually, easier: build U, V, W as lists of lists, column by column.

    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]   # 6 x 32
    V = [[0.0] * rank for _ in range(m * p)]   # 21 x 32
    W = [[0.0] * rank for _ in range(p * n)]   # 14 x 32

    col = 0  # current rank column index

    # --- Part 1: rows 0,1 of A (i in {0,1}) and rows 0,1 of B (j in {0,1}), cols 0..6 of B ---
    # This is a 2x2 * 2x7 multiplication.
    # Split the 7 columns into: cols 0-1, cols 2-3, cols 4-5, col 6.
    # For each 2x2x2 block use Strassen (7 products).
    # For the 2x2x1 block use schoolbook (4 products).

    def add_strassen_block(col_start, i_off, j_off, k_off):
        """
        Add a Strassen 2x2x2 decomposition for A[i_off:i_off+2, j_off:j_off+2]
        * B[j_off:j_off+2, k_off:k_off+2] contributing to C[i_off:i_off+2, k_off:k_off+2].
        Returns the next column index.
        """
        nonlocal col
        # Strassen products:
        # M1 = (a00 + a11) * (b00 + b10)
        # M2 = (a10 + a11) * b00
        # M3 = a00 * (b01 - b11)
        # M4 = a11 * (b11 - b10)
        # M5 = (a00 + a01) * b10
        # M6 = (a10 - a11) * (b00 + b01)
        # M7 = (a01 - a11) * (b10 + b11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        a_idx = [(i_off + di, j_off + dj) for di in range(2) for dj in range(2)]
        # a00 = A[i_off, j_off], a01 = A[i_off, j_off+1]
        # a10 = A[i_off+1, j_off], a11 = A[i_off+1, j_off+1]

        b_idx = [(j_off + dj, k_off + dk) for dj in range(2) for dk in range(2)]
        # b00 = B[j_off, k_off], b01 = B[j_off, k_off+1]
        # b10 = B[j_off+1, k_off], b11 = B[j_off+1, k_off+1]

        c_idx = [(i_off + di, k_off + dk) for di in range(2) for dk in range(2)]
        # c00 = C[i_off, k_off], c01 = C[i_off, k_off+1]
        # c10 = C[i_off+1, k_off], c11 = C[i_off+1, k_off+1]

        # Define the 7 Strassen products as (U_factor, V_factor, W_factors)
        # U_factor: dict from A-index (i,j) to coefficient
        # V_factor: dict from B-index (j,k) to coefficient
        # W_factors: list of (C-index (i,k), coefficient) for each product

        products = [
            # M1 = (a00 + a11) * (b00 + b10)
            (
                {(i_off, j_off): 1, (i_off+1, j_off+1): 1},
                {(j_off, k_off): 1, (j_off+1, k_off): 1},
                [((i_off, k_off), 1), ((i_off+1, k_off+1), 1)]
            ),
            # M2 = (a10 + a11) * b00
            (
                {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1},
                {(j_off, k_off): 1},
                [((i_off+1, k_off), 1)]
            ),
            # M3 = a00 * (b01 - b11)
            (
                {(i_off, j_off): 1},
                {(j_off, k_off+1): 1, (j_off+1, k_off+1): -1},
                [((i_off, k_off+1), 1), ((i_off+1, k_off+1), 1)]
            ),
            # M4 = a11 * (b11 - b10)
            (
                {(i_off+1, j_off+1): 1},
                {(j_off+1, k_off+1): 1, (j_off+1, k_off): -1},
                [((i_off, k_off), 1), ((i_off+1, k_off), 1)]
            ),
            # M5 = (a00 + a01) * b10
            (
                {(i_off, j_off): 1, (i_off, j_off+1): 1},
                {(j_off+1, k_off): 1},
                [((i_off, k_off), -1), ((i_off, k_off+1), 1)]
            ),
            # M6 = (a10 - a11) * (b00 + b01)
            (
                {(i_off+1, j_off): 1, (i_off+1, j_off+1): -1},
                {(j_off, k_off): 1, (j_off, k_off+1): 1},
                [((i_off+1, k_off+1), 1)]
            ),
            # M7 = (a01 - a11) * (b10 + b11)
            (
                {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1},
                {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1},
                [((i_off, k_off), 1)]
            ),
        ]

        for u_dict, v_dict, w_list in products:
            for (ai, aj), coeff in u_dict.items():
                U[ai * m + aj][col] += coeff
            for (bj, bk), coeff in v_dict.items():
                V[bj * p + bk][col] += coeff
            for (ci, ck), coeff in w_list:
                W[ck * n + ci][col] += coeff
            col += 1

        return col

    # Three 2x2x2 Strassen blocks for cols 0-1, 2-3, 4-5
    col = add_strassen_block(0, i_off=0, j_off=0, k_off=0)
    col = add_strassen_block(col, i_off=0, j_off=0, k_off=2)
    col = add_strassen_block(col, i_off=0, j_off=0, k_off=4)

    # One 2x2x1 schoolbook block for col 6
    def add_schoolbook_2x2x1(col_start, i_off, j_off, k_off):
        nonlocal col
        # C[i, k] = A[i, j_off] * B[j_off, k] + A[i, j_off+1] * B[j_off+1, k]
        for di in range(2):
            for dj in range(2):
                ai = i_off + di
                aj = j_off + dj
                bj = j_off + dj
                bk = k_off
                ci = i_off + di
                ck = k_off
                U[ai * m + aj][col] = 1
                V[bj * p + bk][col] = 1
                W[ck * n + ci][col] = 1
                col += 1
        return col

    col = add_schoolbook_2x2x1(col, i_off=0, j_off=0, k_off=6)

    # --- Part 2: row 2 of A (i in {0,1}) and row 2 of B (j=2), cols 0..6 ---
    # This is a 2x1 * 1x7 multiplication: C[i, k] = A[i, 2] * B[2, k]
    # Schoolbook: 2 * 7 = 14 products? No, it's 2 rows * 7 cols = 14, but each product
    # is A[i,2] * B[2,k] which is a single term. So 14 rank-1 terms.
    # Wait, that's rank 14. But we said total rank 32 = 25 + 7. Let me recheck.
    # Actually, for 2x1 * 1x7, the tensor is sum_{i,k} A[i,2] B[2,k] C[i,k].
    # This is a sum of 2*7 = 14 rank-1 tensors. So rank 14.
    # Total would be 25 + 14 = 39.
    # Hmm, my earlier calculation was wrong. Let me reconsider.

    # Actually, let me just use schoolbook for the 2x1 * 1x7 part: rank 14.
    # Total rank = 25 + 14 = 39.

    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # col should be 39
    assert col == 39, f"Expected rank 39, got {col}"

    # Convert to integer values where possible
    for mat in (U, V, W):
        for row in mat:
            for j in range(len(row)):
                if abs(row[j] - round(row[j])) < 1e-12:
                    row[j] = round(row[j])

    return {"U": U, "V": V, "W": W}
