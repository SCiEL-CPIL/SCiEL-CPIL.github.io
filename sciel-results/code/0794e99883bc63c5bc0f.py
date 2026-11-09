"""Matrix multiplication 3x4x4 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]


def _strassen_2x2x2():
    """Return (U, V, W) lists for 2x2 x 2x2 = 2x2, rank 7."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # A index: a00=0, a01=1, a10=2, a11=3
    # B index: b00=0, b01=1, b10=2, b11=3
    # C index (k*2+i): C00=0, C01=1, C10=2, C11=3
    # M1: (a00+a11)(b00+b11)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    # M2: (a10+a11)b00
    U[2][1]=1; U[3][1]=1; V[0][1]=1
    W[2][1]=1; W[3][1]=-1
    # M3: a00(b10-b11)
    U[0][2]=1; V[2][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=1
    # M4: a11(b01-b00)
    U[3][3]=1; V[1][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1
    # M5: (a00+a01)b11
    U[0][4]=1; U[1][4]=1; V[3][4]=1
    W[0][4]=-1; W[1][4]=1
    # M6: (a10-a00)(b00+b01)
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1
    W[3][5]=1
    # M7: (a01-a11)(b10+b11)
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1
    W[0][6]=1
    return U, V, W


def _strip_1x2x2():
    """1x2 x 2x2 = 1x2, rank 4 (schoolbook)."""
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(2)]
    U[0][0]=1; V[0][0]=1; W[0][0]=1
    U[0][1]=1; V[1][1]=1; W[1][1]=1
    U[1][2]=1; V[2][2]=1; W[0][2]=1
    U[1][3]=1; V[3][3]=1; W[1][3]=1
    return U, V, W


def build_rank44(n, m, p):
    """Build the rank-44 block decomposition."""
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    sU, sV, sW = _strassen_2x2x2()

    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        for c in range(7):
            for ai in range(4):
                if sU[ai][c] != 0:
                    U[(i_off + ai // 2) * m + (j_off + ai % 2), col] = sU[ai][c]
            for bi in range(4):
                if sV[bi][c] != 0:
                    V[(j_off + bi // 2) * p + (k_off + bi % 2), col] = sV[bi][c]
            for ci in range(4):
                if sW[ci][c] != 0:
                    W[(k_off + ci // 2) * n + (i_off + ci % 2), col] = sW[ci][c]
            col += 1

    def add_122(i_off, j_off, k_off):
        nonlocal col
        # 1x2 x 2x2 = 1x2, rank 4
        # A is 1x2: indices (i_off, j_off), (i_off, j_off+1)
        # B is 2x2: indices (j_off, k_off), (j_off+1, k_off), (j_off, k_off+1), (j_off+1, k_off+1)
        # C is 1x2: indices (i_off, k_off), (i_off, k_off+1)
        U[i_off * m + (j_off + 0), col] = 1
        V[(j_off + 0) * p + (k_off + 0), col] = 1
        W[(k_off + 0) * n + i_off, col] = 1
        col += 1

        U[i_off * m + (j_off + 1), col] = 1
        V[(j_off + 1) * p + (k_off + 0), col] = 1
        W[(k_off + 0) * n + i_off, col] = 1
        col += 1

        U[i_off * m + (j_off + 0), col] = 1
        V[(j_off + 0) * p + (k_off + 1), col] = 1
        W[(k_off + 1) * n + i_off, col] = 1
        col += 1

        U[i_off * m + (j_off + 1), col] = 1
        V[(j_off + 1) * p + (k_off + 1), col] = 1
        W[(k_off + 1) * n + i_off, col] = 1
        col += 1

    # Top 2x4 block of A (rows 0,1) times B (4x4)
    # Split into 2x2 x 2x2 blocks
    # A_top = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C_top = A_top * B
    # For each 2x2x2 product:
    # A00 * B00 -> C00 (i_off=0, j_off=0, k_off=0)
    # A00 * B01 -> C01 (i_off=0, j_off=0, k_off=2)
    # A01 * B10 -> C00 (i_off=0, j_off=2, k_off=0)
    # A01 * B11 -> C01 (i_off=0, j_off=2, k_off=2)
    # A10 * B00 -> C10 (i_off=1, j_off=0, k_off=0)
    # A10 * B01 -> C11 (i_off=1, j_off=0, k_off=2)
    # A11 * B10 -> C10 (i_off=1, j_off=2, k_off=0)
    # A11 * B11 -> C11 (i_off=1, j_off=2, k_off=2)

    # But we need to accumulate, not just add separate products.
    # The correct approach: C_top = A_top * B where A_top is 2x4 and B is 4x4.
    # We can view this as:
    #   C_top[:, 0:2] = A_top[:, 0:2] * B[0:2, :] + A_top[:, 2:4] * B[2:4, :]
    #   C_top[:, 2:4] = A_top[:, 0:2] * B[0:2, 2:4] + A_top[:, 2:4] * B[2:4, 2:4]
    #
    # Each term is a 2x2 x 2x2 = 2x2 product.
    #
    # Term 1: A_top[:,0:2] (2x2) * B[0:2,0:2] (2x2) -> adds to C_top[:,0:2]
    #   i_off=0, j_off=0, k_off=0
    # Term 2: A_top[:,2:4] (2x2) * B[2:4,0:2] (2x2) -> adds to C_top[:,0:2]
    #   i_off=0, j_off=2, k_off=0
    # Term 3: A_top[:,0:2] (2x2) * B[0:2,2:4] (2x2) -> adds to C_top[:,2:4]
    #   i_off=0, j_off=0, k_off=2
    # Term 4: A_top[:,2:4] (2x2) * B[2:4,2:4] (2x2) -> adds to C_top[:,2:4]
    #   i_off=0, j_off=2, k_off=2

    # Wait, the indices are:
    # A is 3x4: rows 0-2, cols 0-3
    # B is 4x4: rows 0-3, cols 0-3
    # C is 3x4: rows 0-2, cols 0-3
    #
    # A_top is rows 0-1 (2x4)
    # A_bot is row 2 (1x4)
    #
    # For A_top * B:
    #   C_top[i,k] = sum_j A_top[i,j] * B[j,k]
    #   = sum_{j=0}^{1} A_top[i,j] * B[j,k] + sum_{j=2}^{3} A_top[i,j] * B[j,k]
    #
    # First sum: A_top[:,0:2] (2x2) * B[0:2,:] (2x4) -> 2x4
    #   This is two 2x2x2 products:
    #     A_top[:,0:2] * B[0:2,0:2] -> C_top[:,0:2]
    #     A_top[:,0:2] * B[0:2,2:4] -> C_top[:,2:4]
    #
    # Second sum: A_top[:,2:4] (2x2) * B[2:4,:] (2x4) -> 2x4
    #   This is two 2x2x2 products:
    #     A_top[:,2:4] * B[2:4,0:2] -> C_top[:,0:2]
    #     A_top[:,2:4] * B[2:4,2:4] -> C_top[:,2:4]

    # So we need 4 Strassen products for C_top, each contributing to different output blocks.
    # But Strassen gives a rank-7 decomposition of a 2x2x2 tensor.
    # The issue is that we need to ADD the contributions from different terms.
    #
    # The standard approach: use a rank-7 decomposition for each 2x2x2 block,
    # but the total rank is the sum if we just concatenate. That gives 4*7=28 for top.
    #
    # However, we can be smarter. Let's use the block structure more carefully.
    #
    # Actually, the inspiration programs use a different approach:
    # They call add_strassen_222 with specific offsets, which adds 7 rank terms each.
    # Let me check the inspiration code more carefully.
    #
    # From inspiration 1:
    # add_strassen_222(0, 0, 0)  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # add_strassen_222(0, 2, 0)  # A[0:2,2:4] * B[0:2,0:2] -> C[0:2,0:2]
    # add_strassen_222(0, 0, 2)  # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # add_strassen_222(0, 2, 2)  # A[0:2,2:4] * B[0:2,2:4] -> C[0:2,2:4]
    #
    # Wait, that's only 4 calls, each with 7 rank terms = 28 total for the top part.
    # But the signature is add_strassen_222(i_off, j_off, k_off).
    #
    # Looking at the code:
    # U[(i_off+0)*m + (j_off+0), col] = 1  # A[i_off, j_off]
    # U[(i_off+1)*m + (j_off+1), col] = 1  # A[i_off+1, j_off+1]
    # V[(j_off+0)*p + (k_off+0), col] = 1  # B[j_off, k_off]
    # V[(j_off+1)*p + (k_off+1), col] = 1  # B[j_off+1, k_off+1]
    # W[(k_off+0)*n + (i_off+0), col] = 1  # C[i_off, k_off]
    # W[(k_off+1)*n + (i_off+1), col] = 1  # C[i_off+1, k_off+1]
    #
    # This is for M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
    # So i_off is the row offset in A and C, j_off is the col offset in A and row offset in B,
    # k_off is the col offset in B and C.
    #
    # So add_strassen_222(i_off, j_off, k_off) computes the product of
    # A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2]
    # and adds it to C[i_off:i_off+2, k_off:k_off+2].
    #
    # For C_top (rows 0-1 of C):
    # C_top[0:2, 0:2] gets contributions from:
    #   A[0:2, 0:2] * B[0:2, 0:2]  -> add_strassen_222(0, 0, 0)
    #   A[0:2, 2:4] * B[2:4, 0:2]  -> add_strassen_222(0, 2, 0)
    #
    # C_top[0:2, 2:4] gets contributions from:
    #   A[0:2, 0:2] * B[0:2, 2:4]  -> add_strassen_222(0, 0, 2)
    #   A[0:2, 2:4] * B[2:4, 2:4]  -> add_strassen_222(0, 2, 2)
    #
    # That's 4 calls * 7 = 28 rank terms for C_top.
    #
    # For C_bot (row 2 of C):
    # C_bot[2, 0:2] = A[2, 0:2] * B[0:2, 0:2] + A[2, 2:4] * B[2:4, 0:2]
    # C_bot[2, 2:4] = A[2, 0:2] * B[0:2, 2:4] + A[2, 2:4] * B[2:4, 2:4]
    #
    # Each is a 1x2 * 2x2 = 1x2 product, rank 4.
    # So 4 calls * 4 = 16 rank terms for C_bot.
    #
    # Total: 28 + 16 = 44.

    # C_top contributions
    add_strassen_222(0, 0, 0)   # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen_222(0, 2, 0)   # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    add_strassen_222(0, 0, 2)   # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_strassen_222(0, 2, 2)   # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]

    # C_bot contributions
    add_122(2, 0, 0)   # A[2,0:2] * B[0:2,0:2] -> C[2,0:2]
    add_122(2, 2, 0)   # A[2,2:4] * B[2:4,0:2] -> C[2,0:2]
    add_122(2, 0, 2)   # A[2,0:2] * B[0:2,2:4] -> C[2,2:4]
    add_122(2, 2, 2)   # A[2,2:4] * B[2:4,2:4] -> C[2,2:4]

    return U, V, W


def check(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition is valid."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1) > tol:
                    return False
    return True


def try_reduce_to_rank(U, V, W, remove_col, n, m, p, deadline):
    """Try to reduce rank by removing one column and re-solving via ALS."""
    R = U.shape[1] - 1
    U43 = np.delete(U, remove_col, axis=1).copy()
    V43 = np.delete(V, remove_col, axis=1).copy()
    W43 = np.delete(W, remove_col, axis=1).copy()

    # ALS iterations
    for it in range(30):
        if time.time() > deadline - 5:
            break

        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, R))
                for k in range(p):
                    M[k, :] = V43[j * p + k, :] * W43[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U43[i * m + j, :] = sol
                except:
                    pass

        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, R))
                for i in range(n):
                    M[i, :] = U43[i * m + j, :] * W43[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V43[j * p + k, :] = sol
                except:
                    pass

        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, R))
                for j in range(m):
                    M[j, :] = U43[i * m + j, :] * V43[j * p + k, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W43[k * n + i, :] = sol
                except:
                    pass

    # Round to half-integers
    U43 = np.round(U43 * 2) / 2
    V43 = np.round(V43 * 2) / 2
    W43 = np.round(W43 * 2) / 2

    if check(U43, V43, W43, n, m, p):
        return U43, V43, W43

    # Local search: try small perturbations
    for ls_iter in range(50):
        if time.time() > deadline - 2:
            break

        improved = False
        for a in range(n * m):
            for r in range(R):
                old = U43[a, r]
                for delta in [0.5, -0.5]:
                    U43[a, r] = old + delta
                    if check(U43, V43, W43, n, m, p):
                        return U43, V43, W43
                    U43[a, r] = old

        for b in range(m * p):
            for r in range(R):
                old = V43[b, r]
                for delta in [0.5, -0.5]:
                    V43[b, r] = old + delta
                    if check(U43, V43, W43, n, m, p):
                        return U43, V43, W43
                    V43[b, r] = old

        for c in range(p * n):
            for r in range(R):
                old = W43[c, r]
                for delta in [0.5, -0.5]:
                    W43[c, r] = old + delta
                    if check(U43, V43, W43, n, m, p):
                        return U43, V43, W43
                    W43[c, r] = old

    return None


def construct():
    """Build a rank decomposition for 3x4 x 4x4 = 3x4."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    # Build rank-44 solution
    U44, V44, W44 = build_rank44(n, m, p)

    # Verify
    if not check(U44, V44, W44, n, m, p):
        # Fallback to schoolbook
        return _schoolbook()

    # Try to find rank-43 by removing one column and re-solving
    best_result = None
    for col in range(44):
        if time.time() > deadline - 10:
            break
        result = try_reduce_to_rank(U44, V44, W44, col, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Return rank-44
    return {"U": U44.tolist(), "V": V44.tolist(), "W": W44.tolist()}


def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}
