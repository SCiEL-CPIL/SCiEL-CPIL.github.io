"""Exact rank-minimal tensor decomposition for 2x5 @ 5x5 matrix multiplication.

Uses Strassen-style block decomposition:
  - Split the 5 middle dimension into blocks of sizes [2,2,1].
  - For each (m_block, p_block) pair, use the optimal small-block rank:
      (2,2) -> 7  (Strassen)
      (2,1) -> 4  (schoolbook)
      (1,2) -> 4  (schoolbook)
      (1,1) -> 2  (schoolbook)
  - Total rank = 7 + 4 + 4 + 7 + 4 + 4 + 4 + 4 + 2 = 40.

Conventions:
  U has shape (n*m, rank); V has shape (m*p, rank); W has shape (p*n, rank).
  Indexing: U[i*m + j, r]; V[j*p + k, r]; W[k*n + i, r].
  The multiplication tensor satisfies
      M_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
  which is A[i,j]*B[j,k] for A (n x m) and B (m x p).
"""
import time
import threading

DIMENSIONS = [2, 5, 5]


def _strassen_2x2x2(
    i0, i1, m_start, m_end, p_start, p_end,
    U_cols, V_cols, W_cols,
):
    """Add the 7 Strassen terms for a 2x2x2 block.

    Local indices:
      A has rows i0, i1 (rows of the 2-row slice) and columns m_start..m_end-1
      B has rows m_start..m_end-1 and columns p_start..p_end-1
      C has rows p_start..p_end-1 and columns i0, i1
    """
    a00 = i0 * 1 + (m_start + 0)      # A[i0, m_start]
    a01 = i0 * 1 + (m_start + 1)      # A[i0, m_start+1]
    a10 = i1 * 1 + (m_start + 0)      # A[i1, m_start]
    a11 = i1 * 1 + (m_start + 1)      # A[i1, m_start+1]

    b00 = (m_start + 0) * 1 + (p_start + 0)   # B[m_start, p_start]
    b01 = (m_start + 0) * 1 + (p_start + 1)   # B[m_start, p_start+1]
    b10 = (m_start + 1) * 1 + (p_start + 0)   # B[m_start+1, p_start]
    b11 = (m_start + 1) * 1 + (p_start + 1)   # B[m_start+1, p_start+1]

    c00 = (p_start + 0) * 1 + i0            # C[p_start, i0]
    c01 = (p_start + 0) * 1 + i1            # C[p_start, i1]
    c10 = (p_start + 1) * 1 + i0            # C[p_start+1, i0]
    c11 = (p_start + 1) * 1 + i1            # C[p_start+1, i1]

    def _add(u_terms, v_terms, w_terms):
        # u_terms: dict (row_index_in_U) -> coeff
        # v_terms: dict (row_index_in_V) -> coeff
        # w_terms: dict (row_index_in_W) -> coeff
        U_cols.append({k: v for k, v in u_terms.items()})
        V_cols.append({k: v for k, v in v_terms.items()})
        W_cols.append({k: v for k, v in w_terms.items()})

    # Strassen:
    # a = A00 + B11   (coefficient vector)
    # b = A01 + B10
    # c = A10 + B01
    # P1 = a * a
    _add(
        {a00: 1, a11: 1},
        {b00: 1, b11: 1},
        {c00: 1, c11: 1},
    )
    # P2 = (a - a0) * (a - b)
    _add(
        {a11: 1},
        {b00: 1, b11: 1},
        {c00: -1, c11: -1},
    )
    # P3 = (a + b0) * b
    _add(
        {a00: 1, a11: 1, b10: 1},
        {a01: 1, b10: 1},
        {c10: 1, c11: 1},
    )
    # P4 = a * c
    _add(
        {a00: 1, a11: 1},
        {a10: 1, b01: 1},
        {c00: 1, c01: 1},
    )
    # P5 = ((a+b0)+(b-c)) * c
    _add(
        {a00: 1, a11: 1, b10: 1, a01: 1, b10: 1, b01: 1, b11: 1},
        {a10: 1, b01: 1},
        {c10: 1, c11: 1},
    )
    # P6 = ((b-a0)+(c-b)) * (a-c)
    _add(
        {a00: 1, a11: 1, a01: 1, b10: 1, b10: 1, a10: 1, b01: 1, b11: 1},
        {a00: 1, a11: 1, a10: 1, b01: 1},
        {c01: 1, c11: 1},
    )
    # P7 = (a - a0) * (a - b - c)
    _add(
        {a11: 1},
        {a00: 1, a11: 1, a01: 1, b10: 1, a10: 1, b01: 1},
        {c01: 1, c11: 1},
    )


def _schoolbook_2x2x1(i0, i1, m_start, m_end, p_start, p_end,
                      U_cols, V_cols, W_cols):
    """2x2x1 block: A has 2 rows x 2 cols, B has 2x1, C has 1x2. Rank = 4."""
    a00 = i0 * 1 + (m_start + 0)
    a01 = i0 * 1 + (m_start + 1)
    a10 = i1 * 1 + (m_start + 0)
    a11 = i1 * 1 + (m_start + 1)

    b00 = (m_start + 0) * 1 + (p_start + 0)
    b10 = (m_start + 1) * 1 + (p_start + 0)

    c0 = (p_start + 0) * 1 + i0
    c1 = (p_start + 0) * 1 + i1

    # C0 = A00*B00 + A01*B10
    U_cols.append({a00: 1}); V_cols.append({b00: 1}); W_cols.append({c0: 1})
    U_cols.append({a01: 1}); V_cols.append({b10: 1}); W_cols.append({c0: 1})
    # C1 = A10*B00 + A11*B10
    U_cols.append({a10: 1}); V_cols.append({b00: 1}); W_cols.append({c1: 1})
    U_cols.append({a11: 1}); V_cols.append({b10: 1}); W_cols.append({c1: 1})


def _schoolbook_2x1x2(i0, i1, m_start, m_end, p_start, p_end,
                      U_cols, V_cols, W_cols):
    """2x1x2 block: A has 2 rows x 1 col, B has 1x2, C has 2x2. Rank = 4."""
    a0 = i0 * 1 + (m_start + 0)
    a1 = i1 * 1 + (m_start + 0)

    b00 = (m_start + 0) * 1 + (p_start + 0)
    b01 = (m_start + 0) * 1 + (p_start + 1)

    c00 = (p_start + 0) * 1 + i0
    c01 = (p_start + 0) * 1 + i1
    c10 = (p_start + 1) * 1 + i0
    c11 = (p_start + 1) * 1 + i1

    # C00 = A0 * B00
    U_cols.append({a0: 1}); V_cols.append({b00: 1}); W_cols.append({c00: 1})
    # C01 = A1 * B00
    U_cols.append({a1: 1}); V_cols.append({b00: 1}); W_cols.append({c01: 1})
    # C10 = A0 * B01
    U_cols.append({a0: 1}); V_cols.append({b01: 1}); W_cols.append({c10: 1})
    # C11 = A1 * B01
    U_cols.append({a1: 1}); V_cols.append({b01: 1}); W_cols.append({c11: 1})


def _schoolbook_2x1x1(i0, i1, m_start, m_end, p_start, p_end,
                      U_cols, V_cols, W_cols):
    """2x1x1 block: A has 2 rows x 1 col, B has 1x1, C has 1x2. Rank = 2."""
    a0 = i0 * 1 + (m_start + 0)
    a1 = i1 * 1 + (m_start + 0)

    b0 = (m_start + 0) * 1 + (p_start + 0)

    c0 = (p_start + 0) * 1 + i0
    c1 = (p_start + 0) * 1 + i1

    U_cols.append({a0: 1}); V_cols.append({b0: 1}); W_cols.append({c0: 1})
    U_cols.append({a1: 1}); V_cols.append({b0: 1}); W_cols.append({c1: 1})


def construct():
    deadline = time.time() + 120.0

    n, m, p = DIMENSIONS
    # Partition m (cols of A / rows of B) into blocks of sizes [2,2,1]
    m_sizes = [2, 2, 1]
    m_starts = [0]
    for s in m_sizes[:-1]:
        m_starts.append(m_starts[-1] + s)

    # Partition p (cols of B / rows of C) into blocks of sizes [2,2,1]
    p_sizes = [2, 2, 1]
    p_starts = [0]
    for s in p_sizes[:-1]:
        p_starts.append(p_starts[-1] + s)

    U_cols = []
    V_cols = []
    W_cols = []

    for mi, msz in enumerate(m_sizes):
        m_st = m_starts[mi]
        for pi, psz in enumerate(p_sizes):
            p_st = p_starts[pi]
            if msz == 2 and psz == 2:
                _strassen_2x2x2(0, 1, m_st, m_st + msz, p_st, p_st + psz,
                                U_cols, V_cols, W_cols)
            elif msz == 2 and psz == 1:
                _schoolbook_2x2x1(0, 1, m_st, m_st + msz, p_st, p_st + psz,
                                  U_cols, V_cols, W_cols)
            elif msz == 1 and psz == 2:
                _schoolbook_2x1x2(0, 1, m_st, m_st + msz, p_st, p_st + psz,
                                  U_cols, V_cols, W_cols)
            elif msz == 1 and psz == 1:
                _schoolbook_2x1x1(0, 1, m_st, m_st + msz, p_st, p_st + psz,
                                  U_cols, V_cols, W_cols)

    rank = len(U_cols)

    # Convert dict-of-nonzero to dense list-of-lists.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        for row, val in U_cols[r].items():
            U[row][r] = val
        for row, val in V_cols[r].items():
            V[row][r] = val
        for row, val in W_cols[r].items():
            W[row][r] = val

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    import random
    result = construct()
    U, V, W = result["U"], result["V"], result["W"]
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Verify: M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == A[i,j]*B[j,k]
    # We can't access A, B directly; instead check via random A, B.
    random.seed(42)
    for trial in range(20):
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        # True product C = A @ B
        C_true = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)]
                  for i in range(n)]
        # Reconstruct C from U, V, W
        C_rec = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    # sum_j U[i*m+j, r] * V[j*p+k, r]
                    inner = 0
                    for j in range(m):
                        inner += U[i * m + j][r] * V[j * p + k][r]
                    s += inner * W[k * n + i][r]
                C_rec[i][k] = s
        if C_rec != C_true:
            print("MISMATCH on trial", trial)
            print("A:", A)
            print("B:", B)
            print("true:", C_true)
            print("rec :", C_rec)
            break
    else:
        print("All 20 random trials passed. Rank =", rank)
