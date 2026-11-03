"""Matrix multiplication 3x4 x 4x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(r):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        T += u[:, :, None] * v[None, :, :] * w[None, None, :]
    return bool(np.all(np.abs(T - 1.0) <= tol))


def _to_half_integer(arr):
    """Round to nearest half-integer; return list of lists."""
    a = np.asarray(arr, dtype=float)
    return (np.round(a * 2.0) / 2.0).tolist()


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 140.0

    # Block structure:
    #   A rows:  [0:2] (size 2), [2:3] (size 1)
    #   B rows:  [0:2] (size 2), [2:4] (size 2)
    #   B cols:  [0:2], [2:4], [4:6] (each size 2)
    #
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Split j into two halves (size 2 each):
    #   C[i,k] = sum_{j in 0:2} A[i,j]*B[j,k] + sum_{j in 2:4} A[i,j]*B[j,k]
    #
    # For each B-row-half rb (size 2) and B-col-block cb (size 2):
    #   - top A rows (size 2): 2x2 @ 2x2 -> Strassen rank 7
    #   - bottom A row (size 1): 1x2 @ 2x2 -> rank 1
    #
    # Total: (2 rb * 3 cb) * (7 + 1) = 6 * 8 = 48

    U = np.zeros((n * m, 48), dtype=float)
    V = np.zeros((m * p, 48), dtype=float)
    W = np.zeros((p * n, 48), dtype=float)

    col = 0

    def uidx(i, j):
        return i * m + j

    def vidx(j, k):
        return j * p + k

    def widx(k, i):
        return k * n + i

    def add(u_vals, v_vals, w_vals):
        """u_vals: list of (idx, val) for U; v_vals for V; w_vals for W."""
        nonlocal col
        for (idx, val) in u_vals:
            U[idx, col] += val
        for (idx, val) in v_vals:
            V[idx, col] += val
        for (idx, val) in w_vals:
            W[idx, col] += val
        col += 1

    def strassen_2x2(a_rows, b_rows, cb):
        """
        Strassen decomposition for:
            C[a_rows[0]:a_rows[1], cb:cb+2] = A[a_rows[0]:a_rows[1], b_rows[0]:b_rows[1]] @ B[b_rows[0]:b_rows[1], cb:cb+2]
        where a_rows has size 2, b_rows has size 2, cb block has size 2.
        Returns 7 columns added.
        """
        r0, r1 = a_rows   # two A row indices
        b0, b1 = b_rows   # two B row indices
        c0, c1 = cb, cb + 1  # two output col indices

        # Sub-blocks:
        # A00 = A[r0, b0], A01 = A[r0, b1], A10 = A[r1, b0], A11 = A[r1, b1]
        # B00 = B[b0, c0], B01 = B[b0, c1], B10 = B[b1, c0], B11 = B[b1, c1]
        #
        # Strassen:
        # M1 = (A00+A11)(B01+B11)  -> C00 += M1, C11 += M1
        # M2 = (A10+A11)(B00)      -> C10 += M2, C11 -= M2
        # M3 = (A00)(B10-B11)      -> C01 += M3, C11 += M3
        # M4 = (A11)(B00-B10)      -> C00 += M4, C10 += M4
        # M5 = (A00+A10)(B11)      -> C00 -= M5, C01 += M5
        # M6 = (A11-A01)(B00+B10)  -> C11 += M6
        # M7 = (A01-A11)(B10+B11)  -> C00 += M7
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # M1 = (A[r0,b0]+A[r1,b1]) * (B[b0,c1]+B[b1,c1])
        add(
            [(uidx(r0, b0), 1), (uidx(r1, b1), 1)],
            [(vidx(b0, c1), 1), (vidx(b1, c1), 1)],
            [(widx(c0, r0), 1), (widx(c1, r1), 1)],
        )
        # M2 = (A[r1,b0]+A[r1,b1]) * (B[b0,c0])
        add(
            [(uidx(r1, b0), 1), (uidx(r1, b1), 1)],
            [(vidx(b0, c0), 1)],
            [(widx(c0, r1), 1), (widx(c1, r1), -1)],
        )
        # M3 = (A[r0,b0]) * (B[b1,c0]-B[b1,c1])
        add(
            [(uidx(r0, b0), 1)],
            [(vidx(b1, c0), 1), (vidx(b1, c1), -1)],
            [(widx(c1, r0), 1), (widx(c1, r1), 1)],
        )
        # M4 = (A[r1,b1]) * (B[b0,c0]-B[b1,c0])
        add(
            [(uidx(r1, b1), 1)],
            [(vidx(b0, c0), 1), (vidx(b1, c0), -1)],
            [(widx(c0, r0), 1), (widx(c0, r1), 1)],
        )
        # M5 = (A[r0,b0]+A[r1,b0]) * (B[b1,c1])
        add(
            [(uidx(r0, b0), 1), (uidx(r1, b0), 1)],
            [(vidx(b1, c1), 1)],
            [(widx(c0, r0), -1), (widx(c1, r0), 1)],
        )
        # M6 = (A[r1,b1]-A[r0,b1]) * (B[b0,c0]+B[b1,c0])
        add(
            [(uidx(r1, b1), 1), (uidx(r0, b1), -1)],
            [(vidx(b0, c0), 1), (vidx(b1, c0), 1)],
            [(widx(c1, r1), 1)],
        )
        # M7 = (A[r0,b1]-A[r1,b1]) * (B[b1,c0]+B[b1,c1])
        add(
            [(uidx(r0, b1), 1), (uidx(r1, b1), -1)],
            [(vidx(b1, c0), 1), (vidx(b1, c1), 1)],
            [(widx(c0, r0), 1)],
        )

    def rank1_1x2(a_row, b_rows, cb):
        """
        Rank-1 decomposition for:
            C[a_row, cb:cb+2] = A[a_row, b_rows[0]:b_rows[1]] @ B[b_rows[0]:b_rows[1], cb:cb+2]
        where A part is 1x2, B part is 2x2.
        The tensor T[i,j,k] = 1 for all (i in {a_row}, j in b_rows, k in {cb,cb+1}).
        This is rank 1: U[a_row, b] = 1, V[b, k] = 1, W[k, a_row] = 1.
        """
        b0, b1 = b_rows
        c0, c1 = cb, cb + 1
        add(
            [(uidx(a_row, b0), 1), (uidx(a_row, b1), 1)],
            [(vidx(b0, c0), 1), (vidx(b0, c1), 1), (vidx(b1, c0), 1), (vidx(b1, c1), 1)],
            [(widx(c0, a_row), 1), (widx(c1, a_row), 1)],
        )

    # Build the decomposition
    top_rows = [0, 1]
    bot_row = 2
    b_row_blocks = [[0, 1], [2, 3]]
    b_col_blocks = [0, 2, 4]

    for b_rb in b_row_blocks:
        for cb in b_col_blocks:
            # Top 2 A rows: Strassen 2x2 @ 2x2
            strassen_2x2(top_rows, b_rb, cb)
            # Bottom 1 A row: rank-1 1x2 @ 2x2
            rank1_1x2(bot_row, b_rb, cb)

    assert col == 48, f"Expected 48 columns, got {col}"

    # Verify
    if _verify(U, V, W, n, m, p):
        U_out = _to_half_integer(U)
        V_out = _to_half_integer(V)
        W_out = _to_half_integer(W)
        if _verify(U_out, V_out, W_out, n, m, p):
            return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback to schoolbook
    U_s, V_s, W_s = _schoolbook(n, m, p)
    return {"U": U_s, "V": V_s, "W": W_s}
# EVOLVE-BLOCK-END
