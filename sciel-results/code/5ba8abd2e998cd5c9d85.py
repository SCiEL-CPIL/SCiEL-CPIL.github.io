"""Rewritten program: 56-rank 4x4x4 via Strassen blocks, doubled to 112 for 4x4x8."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _build_56_rank_4x4x4():
    """
    Return a rank-56 decomposition of the 4x4x4 multiplication tensor.
    Uses 8 independent Strassen 2x2x2 decompositions (one per block product).
    Returns U4 (16x56), V4 (16x56), W4 (16x56) as float64.
    """
    U4 = np.zeros((16, 56), dtype=np.float64)
    V4 = np.zeros((16, 56), dtype=np.float64)
    W4 = np.zeros((16, 56), dtype=np.float64)

    def strassen_2x2(a_idx, b_idx, c_idx, col_start):
        """Add 7 Strassen terms for 2x2*2x2. Returns next col."""
        col = col_start
        a00, a01, a10, a11 = a_idx
        b00, b01, b10, b11 = b_idx
        c00, c01, c10, c11 = c_idx

        # M1 = (a00+a11)*(b00+b11)
        U4[a00, col] += 1.0; U4[a11, col] += 1.0
        V4[b00, col] += 1.0; V4[b11, col] += 1.0
        W4[c00, col] += 1.0; W4[c11, col] += 1.0
        col += 1

        # M2 = (a10+a11)*b00
        U4[a10, col] += 1.0; U4[a11, col] += 1.0
        V4[b00, col] += 1.0
        W4[c10, col] += 1.0; W4[c11, col] -= 1.0
        col += 1

        # M3 = a00*(b01-b11)
        U4[a00, col] += 1.0
        V4[b01, col] += 1.0; V4[b11, col] -= 1.0
        W4[c01, col] += 1.0; W4[c11, col] += 1.0
        col += 1

        # M4 = a11*(b10-b00)
        U4[a11, col] += 1.0
        V4[b10, col] += 1.0; V4[b00, col] -= 1.0
        W4[c00, col] += 1.0; W4[c10, col] += 1.0
        col += 1

        # M5 = (a00+a01)*b11
        U4[a00, col] += 1.0; U4[a01, col] += 1.0
        V4[b11, col] += 1.0
        W4[c00, col] -= 1.0; W4[c01, col] += 1.0
        col += 1

        # M6 = (a10-a00)*(b00+b01)
        U4[a10, col] += 1.0; U4[a00, col] -= 1.0
        V4[b00, col] += 1.0; V4[b01, col] += 1.0
        W4[c11, col] += 1.0
        col += 1

        # M7 = (a01-a11)*(b10+b11)
        U4[a01, col] += 1.0; U4[a11, col] -= 1.0
        V4[b10, col] += 1.0; V4[b11, col] += 1.0
        W4[c00, col] += 1.0
        col += 1

        return col

    # Block indices for 2x2 submatrices within 4x4
    A00 = [0, 1, 4, 5]
    A01 = [2, 3, 6, 7]
    A10 = [8, 9, 12, 13]
    A11 = [10, 11, 14, 15]

    B00 = [0, 1, 4, 5]
    B01 = [2, 3, 6, 7]
    B10 = [8, 9, 12, 13]
    B11 = [10, 11, 14, 15]

    C00 = [0, 1, 4, 5]
    C01 = [2, 3, 6, 7]
    C10 = [8, 9, 12, 13]
    C11 = [10, 11, 14, 15]

    col = 0
    # C00 = A00*B00 + A01*B10
    col = strassen_2x2(A00, B00, C00, col)
    col = strassen_2x2(A01, B10, C00, col)
    # C01 = A00*B01 + A01*B11
    col = strassen_2x2(A00, B01, C01, col)
    col = strassen_2x2(A01, B11, C01, col)
    # C10 = A10*B00 + A11*B10
    col = strassen_2x2(A10, B00, C10, col)
    col = strassen_2x2(A11, B10, C10, col)
    # C11 = A10*B01 + A11*B11
    col = strassen_2x2(A10, B01, C11, col)
    col = strassen_2x2(A11, B11, C11, col)

    return U4, V4, W4


def _verify(U, V, W, T, tol=1e-9):
    r = U.shape[1]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 4, 4, 8

    # Build the full 4x4x8 multiplication tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Schoolbook decomposition (rank 128)
    rank_schoolbook = n * m * p
    U = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook

    # Try 56-rank 4x4x4 block, doubled to 112 for 4x4x8
    if time.time() < deadline - 5:
        U4, V4, W4 = _build_56_rank_4x4x4()
        rank4 = U4.shape[1]

        if rank4 <= 56:
            U_full = np.zeros((n*m, 2*rank4), dtype=np.float64)
            V_full = np.zeros((m*p, 2*rank4), dtype=np.float64)
            W_full = np.zeros((p*n, 2*rank4), dtype=np.float64)

            # Block 1: k=0..3
            U_full[:, 0:rank4] = U4
            V_full[0:16, 0:rank4] = V4
            W_full[0:16, 0:rank4] = W4

            # Block 2: k=4..7
            U_full[:, rank4:2*rank4] = U4
            V_full[16:32, rank4:2*rank4] = V4
            W_full[16:32, rank4:2*rank4] = W4

            if _verify(U_full, V_full, W_full, T):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = 2*rank4

    # Convert to half-integer lists
    def to_half_integer(mat):
        mat2 = mat * 2
        rounded = np.round(mat2)
        if np.allclose(mat2, rounded, atol=1e-6):
            return (rounded / 2).tolist()
        rounded_int = np.round(mat)
        if np.allclose(mat, rounded_int, atol=1e-6):
            return rounded_int.astype(int).tolist()
        return mat.tolist()

    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
