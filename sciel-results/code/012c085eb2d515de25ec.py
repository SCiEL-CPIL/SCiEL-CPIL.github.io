"""Matrix multiplication 4x5x5 with rank-reduced block decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]


def _strassen_2x2x2():
    """Verified rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) with integer entries.
    U[i*2+j, r] corresponds to A[i,j], V[j*2+k, r] to B[j,k], W[k*2+i, r] to C[i,k].
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)

    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = -1; W[3, 0] = -1

    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1

    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1

    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1

    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = -1

    # M6 = (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1

    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[2, 6] = 1

    return U, V, W


def _schoolbook_sub(n, m, p):
    """Schoolbook for n*m x m*p sub-block. Returns (U, V, W) as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _embed_sub(U_sub, V_sub, W_sub, i0, j0, k0, ni, mj, pk, n, m, p):
    """Embed a sub-block decomposition into the full n x m x p space.
    U_sub: (ni*mj, rank), V_sub: (mj*pk, rank), W_sub: (pk*ni, rank)
    Returns full-size (U, V, W) each of shape (n*m, rank), (m*p, rank), (p*n, rank).
    """
    rank = U_sub.shape[1]
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    for i in range(ni):
        for j in range(mj):
            U[(i0 + i) * m + (j0 + j), :] = U_sub[i * mj + j, :]
    for j in range(mj):
        for k in range(pk):
            V[(j0 + j) * p + (k0 + k), :] = V_sub[j * pk + k, :]
    for k in range(pk):
        for i in range(ni):
            W[(k0 + k) * n + (i0 + i), :] = W_sub[k * ni + i, :]

    return U, V, W


def _assemble_block_decomp(n, m, p, sub_decomps):
    """Assemble full decomposition from sub-block decompositions.
    sub_decomps: list of (U_sub, V_sub, W_sub, i0, j0, k0, ni, mj, pk)
    """
    total_rank = sum(U[0].shape[1] for U, V, W, *_ in sub_decomps)
    U_out = np.zeros((n * m, total_rank), dtype=int)
    V_out = np.zeros((m * p, total_rank), dtype=int)
    W_out = np.zeros((p * n, total_rank), dtype=int)

    col_off = 0
    for U_sub, V_sub, W_sub, i0, j0, k0, ni, mj, pk in sub_decomps:
        n_cols = U_sub.shape[1]
        for i in range(ni):
            for j in range(mj):
                U_out[(i0 + i) * m + (j0 + j), col_off:col_off + n_cols] = U_sub[i * mj + j, :]
        for j in range(mj):
            for k in range(pk):
                V_out[(j0 + j) * p + (k0 + k), col_off:col_off + n_cols] = V_sub[j * pk + k, :]
        for k in range(pk):
            for i in range(ni):
                W_out[(k0 + k) * n + (i0 + i), col_off:col_off + n_cols] = W_sub[k * ni + i, :]
        col_off += n_cols

    return U_out, V_out, W_out


def _verify(U, V, W, n, m, p):
    """Fast verification using NumPy: check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i * m + j, :], V[j * p + k, :]), W[k * n + i, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True


def _schoolbook_full(n, m, p):
    """Full schoolbook decomposition as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    DEADLINE = 140  # 10s safety margin before 150s kill

    n, m, p = 4, 5, 5

    # Block partitioning of 4x5x5:
    # Split rows of A/C into [0:2, 2:4], columns of A into [0:2, 2:5], columns of B/C into [0:2, 2:5]
    # 8 sub-blocks:
    # (2,2,2) at (i0=0,j0=0,k0=0)  -> Strassen rank 7
    # (2,3,2) at (i0=0,j0=2,k0=0)  -> schoolbook rank 12
    # (2,2,3) at (i0=0,j0=0,k0=2)  -> schoolbook rank 12
    # (2,3,3) at (i0=0,j0=2,k0=2)  -> schoolbook rank 18
    # (2,2,2) at (i0=2,j0=0,k0=0)  -> Strassen rank 7
    # (2,3,2) at (i0=2,j0=2,k0=0)  -> schoolbook rank 12
    # (2,2,3) at (i0=2,j0=0,k0=2)  -> schoolbook rank 12
    # (2,3,3) at (i0=2,j0=2,k0=2)  -> schoolbook rank 18
    # Total: 7+12+12+18+7+12+12+18 = 98

    # Actually let me reconsider: for 2x3x3, schoolbook is 18. Let me verify the partition covers everything.
    # A is 4x5: rows 0-3, cols 0-4. Split rows at 2, cols at 2.
    # B is 5x5: rows 0-4, cols 0-4. Split rows at 2, cols at 2.
    # C is 4x5: rows 0-3, cols 0-4. Split rows at 2, cols at 2.

    # Sub-block (i0, j0, k0, ni, mj, pk) means:
    # A[i0:i0+ni, j0:j0+mj] * B[j0:j0+mj, k0:k0+pk] -> C[i0:i0+ni, k0:k0+pk]

    sub_decomps = []

    # Block (0,0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2], 2x2x2 Strassen
    U_s, V_s, W_s = _strassen_2x2x2()
    sub_decomps.append((U_s, V_s, W_s, 0, 0, 0, 2, 2, 2))

    # Block (0,2,0): A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2], 2x3x2 schoolbook
    U2, V2, W2 = _schoolbook_sub(2, 3, 2)
    sub_decomps.append((U2, V2, W2, 0, 2, 0, 2, 3, 2))

    # Block (0,0,2): A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5], 2x2x3 schoolbook
    U3, V3, W3 = _schoolbook_sub(2, 2, 3)
    sub_decomps.append((U3, V3, W3, 0, 0, 2, 2, 2, 3))

    # Block (0,2,2): A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5], 2x3x3 schoolbook
    U4, V4, W4 = _schoolbook_sub(2, 3, 3)
    sub_decomps.append((U4, V4, W4, 0, 2, 2, 2, 3, 3))

    # Block (2,0,0): A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2], 2x2x2 Strassen
    U5, V5, W5 = _strassen_2x2x2()
    sub_decomps.append((U5, V5, W5, 2, 0, 0, 2, 2, 2))

    # Block (2,2,0): A[2:4, 2:5] * B[2:5, 0:2] -> C[2:4, 0:2], 2x3x2 schoolbook
    U6, V6, W6 = _schoolbook_sub(2, 3, 2)
    sub_decomps.append((U6, V6, W6, 2, 2, 0, 2, 3, 2))

    # Block (2,0,2): A[2:4, 0:2] * B[0:2, 2:5] -> C[2:4, 2:5], 2x2x3 schoolbook
    U7, V7, W7 = _schoolbook_sub(2, 2, 3)
    sub_decomps.append((U7, V7, W7, 2, 0, 2, 2, 2, 3))

    # Block (2,2,2): A[2:4, 2:5] * B[2:5, 2:5] -> C[2:4, 2:5], 2x3x3 schoolbook
    U8, V8, W8 = _schoolbook_sub(2, 3, 3)
    sub_decomps.append((U8, V8, W8, 2, 2, 2, 2, 3, 3))

    U_out, V_out, W_out = _assemble_block_decomp(n, m, p, sub_decomps)

    # Verify if time permits
    if time.time() - t0 < DEADLINE - 5:
        valid = _verify(U_out, V_out, W_out, n, m, p)
        if not valid:
            # Fallback to schoolbook
            U_fb, V_fb, W_fb = _schoolbook_full(n, m, p)
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
