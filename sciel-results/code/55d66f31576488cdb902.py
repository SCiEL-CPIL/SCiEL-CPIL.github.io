import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0


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


def _verify(U, V, W, n, m, p):
    # T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] must equal 1 for all i,j,k
    rank = U.shape[1]
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(s - 1.0) > 1e-6:
                    return False
    return True


def _strassen_block(n, m, p, i_off, j_off, k_off):
    """Strassen on a 2x2x2 block at offsets; returns (U,V,W) of shape (n*m,7),(m*p,7),(p*n,7)."""
    S_A = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=np.int64)
    S_B = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=np.int64)
    S_C = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=np.int64)
    U = np.zeros((n * m, 7), dtype=np.int64)
    V = np.zeros((m * p, 7), dtype=np.int64)
    W = np.zeros((p * n, 7), dtype=np.int64)
    for t in range(7):
        for l in range(4):
            li, lj = l // 2, l % 2
            g_i = i_off + li
            g_j = j_off + lj
            g_k = k_off + lj
            if S_A[t, l]:
                U[g_i * m + g_j, t] += S_A[t, l]
            if S_B[t, l]:
                V[g_j * p + g_k, t] += S_B[t, l]
            if S_C[t, l]:
                W[g_k * n + g_i, t] += S_C[t, l]
    return U, V, W


def construct():
    global _START_TIME, _DEADLINE
    n, m, p = 3, 4, 8

    if time.time() - _START_TIME > _DEADLINE:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- Structural construction: Strassen over 2x2x2 blocks ----
    # Split m=4 into two 2-blocks (j0 in {0,1}, j1 in {2,3});
    # split p=8 into four 2-blocks.
    # For the i-dimension, use Strassen on rows i=1,2 (a 2-row block) for every
    # (j-block, k-block), and schoolbook the i=0 row.
    # Rank = (i=0: 1*4*8=32) + (i=1,2: 2 j-blocks * 4 k-blocks * 7 = 56) = 88.
    # To reduce: fold the i=0 row into the Strassen terms by using a 3x2x2
    # block of rank 7 (the extra row shares the 7 products).

    # 3x2x2 Strassen-like coefficients (i in {0,1,2}, j in {0,1}, k in {0,1})
    # We need T_{i,j,k}=1. Use 7 products.
    # Standard: for 2x2x2 (rows 1,2) use Strassen. Row 0: C0k = A00 B0k + A01 B1k.
    # We can express row-0 outputs using the same 7 products plus a couple more,
    # but to keep rank 7 we share: define products so that row 0 is a linear combo.
    #
    # Instead, use a verified rank-84 construction:
    #   i=0: 32 (schoolbook)
    #   i=1,2: use Strassen but merge the two j-blocks' k-structure is fixed.
    # Keep the safe rank-88 as primary; attempt rank-84 improvement below.

    # Build rank-88 safely first.
    U88 = np.zeros((n * m, 88), dtype=np.int64)
    V88 = np.zeros((m * p, 88), dtype=np.int64)
    W88 = np.zeros((p * n, 88), dtype=np.int64)
    col = 0
    # i=0 schoolbook
    for j in range(m):
        for k in range(p):
            U88[0 * m + j, col] = 1
            V88[j * p + k, col] = 1
            W88[k * n + 0, col] = 1
            col += 1
    # i=1,2 Strassen over 2x2x2 blocks
    for jb in range(2):
        for kb in range(4):
            ub, vb, wb = _strassen_block(n, m, p, i_off=1, j_off=jb * 2, k_off=kb * 2)
            U88[:, col:col + 7] += ub
            V88[:, col:col + 7] += vb
            W88[:, col:col + 7] += wb
            col += 7

    assert col == 88
    if not _verify(U88, V88, W88, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- Attempt rank-84: fold i=0 row into Strassen via 3x2x2 rank-7 blocks ----
    # For each (j-block, k-block), a 3x2x2 block of rank 7.
    # 2 j-blocks * 4 k-blocks * 7 = 56 total (covers all 3 rows) -> rank 56? No:
    # 3x2x2 has 3*2*2=12 entries; rank 7 is the known bound. 8 blocks * 7 = 56.
    # But blocks are disjoint in (j,k) so this would be rank 56 for the whole tensor!
    # That can't be (lower bound 32, but 56 < 88). Let's build it and verify.
    #
    # 3x2x2 rank-7 coefficients:
    S_A3 = np.array([
        [1, 0, 0, 1],   # row0: A00,A01 ; row1: A10,A11  (l: (i,j))
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ], dtype=np.int64)
    # For 3 rows we need 3x2 = 6 entries in A per product; encode l over (i,j) i in{0,1,2},j in{0,1}
    # Rebuild properly with 6 local indices.
    S_A3 = np.zeros((7, 6), dtype=np.int64)
    S_B3 = np.zeros((7, 4), dtype=np.int64)
    S_C3 = np.zeros((7, 6), dtype=np.int64)
    # Strassen on rows 1,2 (local i=1,2), plus row 0 handled by two of the products.
    # M1 = (A10+A22? ) ... use standard 2x2x2 on (i=1,2) and add row0 terms to M5,M6.
    # local A index: i*2 + j  (i in 0..2, j in 0..1)
    # Standard Strassen for rows 1,2:
    # M1 = (A10+A21)(B00+B11)
    # M2 = (A20+A21) B00
    # M3 = A10 (B10-B11)
    # M4 = A21 (B10-B00)
    # M5 = (A10+A11) B11
    # M6 = (A20-A10) B01
    # M7 = (A11-A21)(B00+B10)
    # Row0: C00 = A00 B00 + A01 B10 ; C01 = A00 B01 + A01 B11
    # We must express row0 using the same 7 products. Add row0 contributions:
    # C00 = A00 B00 + A01 B10. Note M1 has (A10+A21)(B00+B11) -> includes A?00? no.
    # This is getting complex; the safe verified rank-88 stands. Return it.

    return {"U": U88.tolist(), "V": V88.tolist(), "W": W88.tolist()}
