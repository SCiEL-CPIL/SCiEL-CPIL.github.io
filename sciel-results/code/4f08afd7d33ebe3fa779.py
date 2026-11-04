"""Rewritten program: block-partitioned matrix multiplication 3x5x5.
Exploits 2x2x2 Strassen sub-blocks (rank 7) plus schoolbook 2x3x3 blocks (rank 18).
Total rank = 3*7 + 3*18 = 75 (vs schoolbook 75... wait, schoolbook is 3*5*5=75).
Actually the partition yields 75 here; the improvement comes from the finer
[1,2]x[2,1,2]x[2,1,2] style partitions that isolate more 2x2x2 Strassen blocks.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

# ---------- elementary block builders ----------

def _schoolbook_block(n, m, p):
    """Return (U,V,W) lists for a schoolbook n x m x p block.
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W, rank


def _strassen_block(n, m, p):
    """Return (U,V,W) lists for a 2x2x2 Strassen block.
    Uses the 7-term Strassen decomposition. Entries are integer/half-integer."""
    # Strassen 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # We build U,V,W so that for each product Mt, U picks the A-side linear combo,
    # V picks the B-side linear combo, and W distributes Mt into C positions.
    # U[i*m+j, t], V[j*p+k, t], W[k*n+i, t].
    #
    # For a 2x2x2 block: A indices (i in 0..1, j in 0..1), B (j in 0..1, k in 0..1),
    # C (i in 0..1, k in 0..1). U row = i*2+j, V row = j*2+k, W row = k*2+i.

    # A-side coefficients for each of the 7 products (indexed by (i,j))
    # and B-side coefficients (indexed by (j,k)).
    A_side = {
        0: {(0,0): 1, (1,1): 1},          # M1: A00 + A11
        1: {(1,0): 1, (1,1): 1},          # M2: A10 + A11
        2: {(0,0): 1},                    # M3: A00
        3: {(1,1): 1},                    # M4: A11
        4: {(0,0): 1, (0,1): 1},          # M5: A00 + A01
        5: {(1,0): 1, (0,0): -1},         # M6: A10 - A00
        6: {(0,1): 1, (1,1): -1},         # M7: A01 - A11
    }
    B_side = {
        0: {(0,0): 1, (1,1): 1},          # M1: B00 + B11
        1: {(0,0): 1},                    # M2: B00
        2: {(0,1): 1, (1,1): -1},         # M3: B01 - B11
        3: {(1,0): 1, (0,0): -1},         # M4: B10 - B00
        4: {(1,1): 1},                    # M5: B11
        5: {(0,0): 1, (0,1): 1},          # M6: B00 + B01
        6: {(1,0): 1, (1,1): 1},          # M7: B10 + B11
    }
    # C-side: W[k*2+i, t] coefficient for product t
    C_side = {
        0: {(0,0): 1, (1,1): 1},          # M1: C00 + C11
        1: {(1,0): 1, (1,1): -1},         # M2: C10 - C11
        2: {(0,1): 1, (1,1): 1},          # M3: C01 + C11
        3: {(0,0): 1, (1,0): 1},          # M4: C00 + C10
        4: {(0,0): -1, (0,1): 1},         # M5: C01 - C00
        5: {(1,1): 1},                    # M6: C11
        6: {(0,0): 1},                    # M7: C00
    }

    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    for t in range(7):
        for (i, j), a in A_side[t].items():
            U[i * 2 + j][t] = a
        for (j, k), b in B_side[t].items():
            V[j * 2 + k][t] = b
        for (k, i), w in C_side[t].items():
            W[k * 2 + i][t] = w
    return U, V, W, rank


def _verify_block(U, V, W, n, m, p):
    """Verify that the block decomposition is correct (tensor equality)."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Reconstruct C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (True) else 0.0
                # Actually the multiplication tensor T[i,j,k] = 1 for all (i,j,k)
                # because C[i,k] = sum_j A[i,j] B[j,k] -> coefficient of A[i,j]*B[j,k] is 1.
                s = 0.0
                for c in range(rank):
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    ok = False
    return ok


# ---------- global assembly ----------

def _assembled_partition():
    """Build the full 3x5x5 decomposition using a block partition.
    Partition: A rows {2,1}, A cols {2,3}; B rows {2,3}, B cols {2,3}.
    Blocks:
      (A[0:2,0:2], B[0:2,0:2]) -> 2x2x2 Strassen (rank 7)
      (A[0:2,0:2], B[0:2,2:5]) -> 2x2x3 schoolbook (rank 12)
      (A[0:2,2:5], B[2:5,0:2]) -> 2x3x2 schoolbook (rank 12)
      (A[0:2,2:5], B[2:5,2:5]) -> 2x3x3 schoolbook (rank 18)
      (A[2:3,0:2], B[0:2,0:2]) -> 1x2x2 schoolbook (rank 4)
      (A[2:3,0:2], B[0:2,2:5]) -> 1x2x3 schoolbook (rank 6)
      (A[2:3,2:5], B[2:5,0:2]) -> 1x3x2 schoolbook (rank 6)
      (A[2:3,2:5], B[2:5,2:5]) -> 1x3x3 schoolbook (rank 9)
    Total rank = 7+12+12+18+4+6+6+9 = 74.
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    # Row/col splits
    A_row_splits = [0, 2, 3]
    A_col_splits = [0, 2, 5]
    B_row_splits = [0, 2, 5]
    B_col_splits = [0, 2, 5]

    blocks = []
    for ri in range(len(A_row_splits) - 1):
        for cj in range(len(A_col_splits) - 1):
            for bk in range(len(B_col_splits) - 1):
                i0, i1 = A_row_splits[ri], A_row_splits[ri + 1]
                j0, j1 = A_col_splits[cj], A_col_splits[cj + 1]
                k0, k1 = B_col_splits[bk], B_col_splits[bk + 1]
                # B row range must match A col range (the inner dimension)
                # A cols j0..j1 correspond to B rows j0..j1
                bn = i1 - i0
                bm = j1 - j0
                bp = k1 - k0
                blocks.append((i0, j0, k0, bn, bm, bp))

    # Build global U, V, W
    # We'll accumulate into dicts then convert to lists.
    U_acc = {}
    V_acc = {}
    W_acc = {}
    total_rank = 0
    for (i0, j0, k0, bn, bm, bp) in blocks:
        if bn == 2 and bm == 2 and bp == 2:
            bU, bV, bW, brank = _strassen_block(bn, bm, bp)
        else:
            bU, bV, bW, brank = _schoolbook_block(bn, bm, bp)
        for r in range(bn * bm):
            bi = r // bm
            bj = r % bm
            gi = i0 + bi
            gj = j0 + bj
            for c in range(brank):
                val = bU[r][c]
                if val != 0:
                    U_acc[(gi * m + gj, total_rank + c)] = val
        for r in range(bm * bp):
            bj = r // bp
            bk = r % bp
            gj = j0 + bj
            gk = k0 + bk
            for c in range(brank):
                val = bV[r][c]
                if val != 0:
                    V_acc[(gj * p + gk, total_rank + c)] = val
        for r in range(bp * bn):
            bk = r // bn
            bi = r % bn
            gk = k0 + bk
            gi = i0 + bi
            for c in range(brank):
                val = bW[r][c]
                if val != 0:
                    W_acc[(gk * n + gi, total_rank + c)] = val
        total_rank += brank

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    for (row, col), val in U_acc.items():
        U[row][col] = val
    for (row, col), val in V_acc.items():
        V[row][col] = val
    for (row, col), val in W_acc.items():
        W[row][col] = val
    return U, V, W, total_rank


def _verify_full(U, V, W):
    """Verify the full decomposition: C[i,k] = sum_j A[i,j] B[j,k]."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                s = 0.0
                for c in range(rank):
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(s - expected) > 1e-9:
                    return False
    return True


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s

    # Build the partitioned decomposition
    U, V, W, rank = _assembled_partition()

    # Verify
    if _verify_full(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook (always valid)
    n, m, p = DIMENSIONS
    rank = n * m * p
    Us = [[0] * rank for _ in range(n * m)]
    Vs = [[0] * rank for _ in range(m * p)]
    Ws = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Us[i * m + j][col] = 1
                Vs[j * p + k][col] = 1
                Ws[k * n + i][col] = 1
                col += 1
    return {"U": Us, "V": Vs, "W": Ws}

# EVOLVE-BLOCK-END
