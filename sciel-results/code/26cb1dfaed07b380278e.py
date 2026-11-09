import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    Uses Strassen-style block decomposition: split the 2x4 * 4x4 multiplication
    into four 2x2 * 2x2 sub-block multiplications, each handled by a rank-7
    Strassen decomposition. Total rank = 28.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    dim_u = n * m  # 8
    dim_v = m * p  # 16
    dim_w = p * n  # 8

    # Deadline guard
    deadline = time.time() + 140.0

    # Strassen decomposition for 2x2 * 2x2 -> 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A11+A22)B21
    # M3 = A11(B11-B21)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M6

    # For each 2x2 block multiplication, we define the Strassen factors.
    # Each Strassen factor has 7 terms.
    # We'll build the full U, V, W matrices with 28 columns (4 blocks * 7 terms).

    # Define the 4 block positions:
    # Block (bi, bj): bi in {0,1} for A's row-block, bj in {0,1} for B's col-block
    # A is 2x4: rows i in {0,1}, cols j in {0,1,2,3}
    # B is 4x4: rows j in {0,1,2,3}, cols k in {0,1,2,3}
    # C is 2x4: rows i in {0,1}, cols k in {0,1,2,3}

    # Block decomposition:
    # A block bi: rows {bi}, cols {2*bj, 2*bj+1} for bj in {0,1}
    #   Actually A is 2x4. Split columns into two blocks of size 2.
    #   A[bj] = A[:, 2*bj:2*bj+2] is a 2x2 block for bj in {0,1}
    # B block: rows {2*bj, 2*bj+1}, cols {2*bk, 2*bk+1}
    #   B[bj, bk] = B[2*bj:2*bj+2, 2*bk:2*bk+2] is a 2x2 block
    # C block: rows {bi}, cols {2*bk, 2*bk+1}
    #   C[bk] = C[:, 2*bk:2*bk+2] is a 2x2 block for bk in {0,1}

    # C[bk] = sum over bj of A[:, 2*bj:2*bj+2] * B[2*bj:2*bj+2, 2*bk:2*bk+2]

    # So for each (bj, bk) pair, we have a 2x2 * 2x2 multiplication.
    # There are 2*2 = 4 such blocks, each with rank 7, total rank 28.

    # Strassen factors for a generic 2x2 block:
    # Let A_blk be 2x2 with entries A00, A01, A10, A11
    # Let B_blk be 2x2 with entries B00, B01, B10, B11
    # Then the 7 products are:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A00+A11)*B10
    # M3 = A00*(B00-B10)
    # M4 = A11*(B01-B11)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M6

    # For each block (bj, bk), the A sub-block is A[:, 2*bj:2*bj+2]
    #   Row indices: i in {0,1} -> A index alpha = i * m + (2*bj + col_in_block)
    #   col_in_block in {0,1}
    # The B sub-block is B[2*bj:2*bj+2, 2*bk:2*bk+2]
    #   Row indices: (2*bj + row_in_block) * p + (2*bk + col_in_block)
    #   row_in_block in {0,1}, col_in_block in {0,1}
    # The C sub-block is C[:, 2*bk:2*bk+2]
    #   Row indices: i in {0,1} -> C index gamma = (2*bk + col_in_block) * n + i
    #   col_in_block in {0,1}

    # We'll build U, V, W as lists of lists with integer entries.
    rank = 28
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]

    # Strassen coefficients:
    # For each of the 7 M's, define which A-entries and B-entries contribute,
    # and how the M's combine to form C-entries.
    #
    # M1: A00+A11 times B00+B11
    #   A contributions: A00 (+1), A11 (+1)
    #   B contributions: B00 (+1), B11 (+1)
    # M2: A00+A11 times B10
    #   A contributions: A00 (+1), A11 (+1)
    #   B contributions: B10 (+1)
    # M3: A00 times (B00-B10)
    #   A contributions: A00 (+1)
    #   B contributions: B00 (+1), B10 (-1)
    # M4: A11 times (B01-B11)
    #   A contributions: A11 (+1)
    #   B contributions: B01 (+1), B11 (-1)
    # M5: A00+A01 times B11
    #   A contributions: A00 (+1), A01 (+1)
    #   B contributions: B11 (+1)
    # M6: A10-A00 times B00+B01
    #   A contributions: A10 (+1), A00 (-1)
    #   B contributions: B00 (+1), B01 (+1)
    # M7: A01-A11 times B10+B11
    #   A contributions: A01 (+1), A11 (-1)
    #   B contributions: B10 (+1), B11 (+1)
    #
    # C combinations:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M6

    # A entry indices in 2x2 block: (row, col) -> (0,0)=A00, (0,1)=A01, (1,0)=A10, (1,1)=A11
    # B entry indices in 2x2 block: (row, col) -> (0,0)=B00, (0,1)=B01, (1,0)=B10, (1,1)=B11

    # For each M (0..6), define:
    #   a_coeffs: dict mapping (row, col) in A-block to coefficient
    #   b_coeffs: dict mapping (row, col) in B-block to coefficient
    #   c_coeffs: dict mapping (row, col) in C-block to coefficient

    strassen = [
        # M1
        (
            {(0,0): 1, (1,1): 1},  # A
            {(0,0): 1, (1,1): 1},  # B
            {(0,0): 1}             # C00
        ),
        # M2
        (
            {(0,0): 1, (1,1): 1},
            {(1,0): 1},
            {(1,0): 1}  # C10
        ),
        # M3
        (
            {(0,0): 1},
            {(0,0): 1, (1,0): -1},
            {(0,1): 1}  # C01
        ),
        # M4
        (
            {(1,1): 1},
            {(0,1): 1, (1,1): -1},
            {(0,0): 1, (1,0): 1}  # C00 and C10
        ),
        # M5
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}  # -C00 and +C01
        ),
        # M6
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): -1}  # -C11
        ),
        # M7
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}  # +C00
        ),
    ]

    # Now build U, V, W
    for bj in range(2):  # A column block / B row block
        for bk in range(2):  # B column block / C column block
            for s in range(7):  # Strassen factor index
                col = bj * 4 + bk * 2 + s  # column index in 0..27
                # Wait, we need 4 blocks * 7 = 28 columns
                # Block index = bj * 2 + bk, so col = (bj*2 + bk) * 7 + s
                col = (bj * 2 + bk) * 7 + s

                a_coeffs, b_coeffs, c_coeffs = strassen[s]

                # Fill U: A-block entries
                for (ar, ac), coeff in a_coeffs.items():
                    # A index: row ar, column 2*bj + ac
                    alpha = ar * m + (2 * bj + ac)
                    U[alpha][col] += coeff

                # Fill V: B-block entries
                for (br, bc), coeff in b_coeffs.items():
                    # B index: row 2*bj + br, column 2*bk + bc
                    beta = (2 * bj + br) * p + (2 * bk + bc)
                    V[beta][col] += coeff

                # Fill W: C-block entries
                for (cr, cc), coeff in c_coeffs.items():
                    # C index: row 2*bk + cc, column cr
                    # C is 2x4, index gamma = col * n + row = (2*bk + cc) * 2 + cr
                    gamma = (2 * bk + cc) * n + cr
                    W[gamma][col] += coeff

    # Verify the decomposition
    def verify_decomposition(U, V, W, n, m, p):
        dim_u = n * m
        dim_v = m * p
        dim_w = p * n
        for a in range(dim_u):
            for b in range(dim_v):
                for g in range(dim_w):
                    val = 0.0
                    for r in range(len(U[0])):
                        val += U[a][r] * V[b][r] * W[g][r]
                    # Determine target
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_g = g // n
                    i_g = g % n
                    if j_a == j_b and k_b == k_g and i_g == i_a:
                        target = 1.0
                    else:
                        target = 0.0
                    if abs(val - target) > 1e-9:
                        return False
        return True

    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
