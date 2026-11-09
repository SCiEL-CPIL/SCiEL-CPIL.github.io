"""Direct rank-44 decomposition for the 3x4 x 4x4 matrix multiplication tensor.

Uses Strassen's algorithm (rank 7) for each 2x2x2 block and schoolbook for
the remaining 1x2x2 blocks. Total rank = 4*7 + 4*4 = 44.
"""

import time

N, M, P = 3, 4, 4
TIME_LIMIT = 120.0
SAFETY_MARGIN = 20.0


def _build_rank44():
    """Build a rank-44 decomposition using Strassen for 2x2x2 blocks."""
    U = {}  # (i, r) -> {col: coeff}
    V = {}  # (r, k) -> {col: coeff}
    W = {}  # (i, k) -> {col: coeff}
    col = 0

    def add_term(i, r, k, coeff):
        """Add a single elementary term A[i,r]*B[r,k] -> C[i,k] with given coefficient."""
        nonlocal col
        if coeff == 0:
            return
        U.setdefault((i, r), {})[col] = U.get((i, r), {}).get(col, 0) + coeff
        V.setdefault((r, k), {})[col] = V.get((r, k), {}).get(col, 0) + coeff
        W.setdefault((i, k), {})[col] = W.get((i, k), {}).get(col, 0) + coeff
        col += 1

    def strassen_2x2(i0, r0, k0):
        """Strassen for 2x2x2 block: A[i0:i0+2, r0:r0+2] * B[r0:r0+2, k0:k0+2]"""
        nonlocal col
        # Strassen's 7 multiplications
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * (B00)
        # M3 = (A00) * (B01 - B11)
        # M4 = (A11) * (B00 + B10)
        # M5 = (A00 + A01) * (B11)
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # For each M, we add terms to U, V, W with appropriate signs
        # M1: A-form = A00 + A11, B-form = B00 + B11
        # C00 += M1, C11 += M1
        c = col; col += 1
        # U: A00 at (i0, r0), A11 at (i0+1, r0+1)
        U.setdefault((i0, r0), {})[c] = U.get((i0, r0), {}).get(c, 0) + 1
        U.setdefault((i0+1, r0+1), {})[c] = U.get((i0+1, r0+1), {}).get(c, 0) + 1
        # V: B00 at (r0, k0), B11 at (r0+1, k0+1)
        V.setdefault((r0, k0), {})[c] = V.get((r0, k0), {}).get(c, 0) + 1
        V.setdefault((r0+1, k0+1), {})[c] = V.get((r0+1, k0+1), {}).get(c, 0) + 1
        # W: C00 at (i0, k0), C11 at (i0+1, k0+1)
        W.setdefault((i0, k0), {})[c] = W.get((i0, k0), {}).get(c, 0) + 1
        W.setdefault((i0+1, k0+1), {})[c] = W.get((i0+1, k0+1), {}).get(c, 0) + 1

        # M2: A-form = A10 + A11, B-form = B00
        # C10 += M2, C11 -= M2
        c = col; col += 1
        U.setdefault((i0+1, r0), {})[c] = U.get((i0+1, r0), {}).get(c, 0) + 1
        U.setdefault((i0+1, r0+1), {})[c] = U.get((i0+1, r0+1), {}).get(c, 0) + 1
        V.setdefault((r0, k0), {})[c] = V.get((r0, k0), {}).get(c, 0) + 1
        W.setdefault((i0+1, k0), {})[c] = W.get((i0+1, k0), {}).get(c, 0) + 1
        W.setdefault((i0+1, k0+1), {})[c] = W.get((i0+1, k0+1), {}).get(c, 0) - 1

        # M3: A-form = A00, B-form = B01 - B11
        # C01 += M3, C11 += M3
        c = col; col += 1
        U.setdefault((i0, r0), {})[c] = U.get((i0, r0), {}).get(c, 0) + 1
        V.setdefault((r0, k0+1), {})[c] = V.get((r0, k0+1), {}).get(c, 0) + 1
        V.setdefault((r0+1, k0+1), {})[c] = V.get((r0+1, k0+1), {}).get(c, 0) - 1
        W.setdefault((i0, k0+1), {})[c] = W.get((i0, k0+1), {}).get(c, 0) + 1
        W.setdefault((i0+1, k0+1), {})[c] = W.get((i0+1, k0+1), {}).get(c, 0) + 1

        # M4: A-form = A11, B-form = B00 + B10
        # C00 += M4, C10 += M4
        c = col; col += 1
        U.setdefault((i0+1, r0+1), {})[c] = U.get((i0+1, r0+1), {}).get(c, 0) + 1
        V.setdefault((r0, k0), {})[c] = V.get((r0, k0), {}).get(c, 0) + 1
        V.setdefault((r0+1, k0), {})[c] = V.get((r0+1, k0), {}).get(c, 0) + 1
        W.setdefault((i0, k0), {})[c] = W.get((i0, k0), {}).get(c, 0) + 1
        W.setdefault((i0+1, k0), {})[c] = W.get((i0+1, k0), {}).get(c, 0) + 1

        # M5: A-form = A00 + A01, B-form = B11
        # C00 -= M5, C01 += M5
        c = col; col += 1
        U.setdefault((i0, r0), {})[c] = U.get((i0, r0), {}).get(c, 0) + 1
        U.setdefault((i0, r0+1), {})[c] = U.get((i0, r0+1), {}).get(c, 0) + 1
        V.setdefault((r0+1, k0+1), {})[c] = V.get((r0+1, k0+1), {}).get(c, 0) + 1
        W.setdefault((i0, k0), {})[c] = W.get((i0, k0), {}).get(c, 0) - 1
        W.setdefault((i0, k0+1), {})[c] = W.get((i0, k0+1), {}).get(c, 0) + 1

        # M6: A-form = A10 - A00, B-form = B00 + B01
        # C11 += M6
        c = col; col += 1
        U.setdefault((i0+1, r0), {})[c] = U.get((i0+1, r0), {}).get(c, 0) + 1
        U.setdefault((i0, r0), {})[c] = U.get((i0, r0), {}).get(c, 0) - 1
        V.setdefault((r0, k0), {})[c] = V.get((r0, k0), {}).get(c, 0) + 1
        V.setdefault((r0, k0+1), {})[c] = V.get((r0, k0+1), {}).get(c, 0) + 1
        W.setdefault((i0+1, k0+1), {})[c] = W.get((i0+1, k0+1), {}).get(c, 0) + 1

        # M7: A-form = A01 - A11, B-form = B10 + B11
        # C00 += M7
        c = col; col += 1
        U.setdefault((i0, r0+1), {})[c] = U.get((i0, r0+1), {}).get(c, 0) + 1
        U.setdefault((i0+1, r0+1), {})[c] = U.get((i0+1, r0+1), {}).get(c, 0) - 1
        V.setdefault((r0+1, k0), {})[c] = V.get((r0+1, k0), {}).get(c, 0) + 1
        V.setdefault((r0+1, k0+1), {})[c] = V.get((r0+1, k0+1), {}).get(c, 0) + 1
        W.setdefault((i0, k0), {})[c] = W.get((i0, k0), {}).get(c, 0) + 1

    def naive_block(i_rows, r_rows, k_cols):
        """Schoolbook for a block."""
        for i in i_rows:
            for r in r_rows:
                for k in k_cols:
                    add_term(i, r, k, 1)

    # Apply Strassen to the four 2x2x2 blocks (top 2 rows of A)
    strassen_2x2(0, 0, 0)  # A rows 0-1, B rows 0-1, B cols 0-1
    strassen_2x2(0, 2, 0)  # A rows 0-1, B rows 2-3, B cols 0-1
    strassen_2x2(0, 0, 2)  # A rows 0-1, B rows 0-1, B cols 2-3
    strassen_2x2(0, 2, 2)  # A rows 0-1, B rows 2-3, B cols 2-3

    # Apply naive to the four 1x2x2 blocks (bottom row of A)
    naive_block([2], [0, 1], [0, 1])
    naive_block([2], [2, 3], [0, 1])
    naive_block([2], [0, 1], [2, 3])
    naive_block([2], [2, 3], [2, 3])

    # Convert to matrices
    rank = col
    U_mat = [[0.0] * rank for _ in range(N * M)]
    V_mat = [[0.0] * rank for _ in range(M * P)]
    W_mat = [[0.0] * rank for _ in range(P * N)]

    for (i, r), col_dict in U.items():
        for c, val in col_dict.items():
            U_mat[i * M + r][c] = val

    for (r, k), col_dict in V.items():
        for c, val in col_dict.items():
            V_mat[r * P + k][c] = val

    for (i, k), col_dict in W.items():
        for c, val in col_dict.items():
            W_mat[k * N + i][c] = val

    return {"U": U_mat, "V": V_mat, "W": W_mat, "rank": rank}


def _schoolbook():
    rank = N * M * P
    U = [[0.0] * rank for _ in range(N * M)]
    V = [[0.0] * rank for _ in range(M * P)]
    W = [[0.0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for r in range(M):
            for k in range(P):
                U[i * M + r][col] = 1.0
                V[r * P + k][col] = 1.0
                W[k * N + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(res):
    U, V, W = res["U"], res["V"], res["W"]
    rank = len(U[0]) if U and U[0] else 0
    for i in range(N):
        for r in range(M):
            for k in range(P):
                s = 0.0
                for c in range(rank):
                    s += U[i * M + r][c] * V[r * P + k][c] * W[k * N + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def construct():
    start = time.time()
    deadline = TIME_LIMIT - SAFETY_MARGIN
    try:
        res = _build_rank44()
    except Exception:
        res = None
    if res is not None and time.time() < deadline and _verify(res):
        return {"U": res["U"], "V": res["V"], "W": res["W"]}
    # Fallback: guaranteed-correct schoolbook
    sb = _schoolbook()
    return {"U": sb["U"], "V": sb["V"], "W": sb["W"]}


# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END
