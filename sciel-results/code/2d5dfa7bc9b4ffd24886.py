# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _MARGIN


def _verify(U, V, W, tol=1e-7):
    """Check exact decomposition: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all i,j,k."""
    n, m, p = 3, 3, 3
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook():
    n, m, p = 3, 3, 3
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2_block():
    """
    Returns U, V, W for a rank-7 Strassen decomposition of 2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    n, m, p = 2, 2, 2
    U = np.zeros((n * m, 7))
    V = np.zeros((m * p, 7))
    W = np.zeros((p * n, 7))

    # Strassen's 7 products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 - P4 + P6
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P3 + P7

    # P1: (A00+A11)(B00+B11)
    U[0, 0] = 1.0; U[3, 0] = 1.0  # A00, A11
    V[0, 0] = 1.0; V[3, 0] = 1.0  # B00, B11
    W[0, 0] = 1.0  # C00
    W[3, 0] = 1.0  # C11

    # P2: (A10+A11)B00
    U[2, 1] = 1.0; U[3, 1] = 1.0  # A10, A11
    V[0, 1] = 1.0  # B00
    W[2, 1] = 1.0  # C10

    # P3: A00(B01-B11)
    U[0, 2] = 1.0  # A00
    V[1, 2] = 1.0; V[3, 2] = -1.0  # B01, -B11
    W[1, 2] = 1.0  # C01
    W[3, 2] = -1.0  # C11

    # P4: A11(B10-B00)
    U[3, 3] = 1.0  # A11
    V[2, 3] = 1.0; V[0, 3] = -1.0  # B10, -B00
    W[0, 3] = -1.0  # C00
    W[2, 3] = 1.0  # C10

    # P5: (A00+A01)B11
    U[0, 4] = 1.0; U[1, 4] = 1.0  # A00, A01
    V[3, 4] = 1.0  # B11
    W[1, 4] = 1.0  # C01

    # P6: (A10-A00)(B00+B01)
    U[2, 5] = 1.0; U[0, 5] = -1.0  # A10, -A00
    V[0, 5] = 1.0; V[1, 5] = 1.0  # B00, B01
    W[0, 5] = 1.0  # C00

    # P7: (A01-A11)(B10+B11)
    U[1, 6] = 1.0; U[3, 6] = -1.0  # A01, -A11
    V[2, 6] = 1.0; V[3, 6] = 1.0  # B10, B11
    W[3, 6] = 1.0  # C11

    return U, V, W


def _rank26_decomposition():
    """
    Construct a rank-26 decomposition for 3x3 matrix multiplication.
    Partition into 2x2 block (top-left) + 1x1 (bottom-right) + cross terms.
    Use Strassen for the 2x2 block (rank 7).
    Cross terms and the 1x1 term use schoolbook-style (rank 1 each).
    """
    n, m, p = 3, 3, 3
    rank = 26
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # First 7 columns: Strassen for 2x2 block (indices 0,1 for i,j,k in block)
    U2, V2, W2 = _strassen_2x2_block()
    U[:, :7] = U2
    V[:, :7] = V2
    W[:, :7] = W2

    # Now add cross terms and 1x1 term.
    # The 2x2 block handles C[0:2, 0:2] from A[0:2, 0:2] * B[0:2, 0:2].
    # We need to handle:
    # 1. C[0:2, 2] = A[0:2, 2] * B[2, 2] + A[0:2, 0:2] * B[0:2, 2]
    #    But the 2x2 block already handles A[0:2,0:2]*B[0:2,0:2] for C[0:2,0:2].
    #    So we need separate terms for:
    #    - A[0:2, 2] * B[2, 2] -> contributes to C[0:2, 2]
    #    - A[0:2, 0:2] * B[0:2, 2] -> contributes to C[0:2, 2]
    #    - A[2, 0:2] * B[0:2, 2] -> contributes to C[2, 2]
    #    - A[2, 2] * B[2, 2] -> contributes to C[2, 2]
    #    - A[0:2, 0:2] * B[0:2, 0] -> contributes to C[0:2, 0:2] (already handled by Strassen? No, Strassen handles B[0:2, 0:2])
    #
    # Actually, let's think more carefully.
    # The full 3x3 multiplication is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Partition:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1
    #
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #      [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    #
    # The 2x2 block Strassen handles A11*B11 (rank 7).
    # Remaining terms:
    # - A12*B21: 2x1 * 1x2 = 2x2 (rank 4 schoolbook, but can be done in rank 4)
    #   Actually A12 is 2x1, B21 is 1x2, so product is 2x2. Each entry is A12[j]*B21[k].
    #   This is outer product, rank 1 if we factor it? No, it's a sum of products.
    #   A12*B21 = [A12[0]*B21[0], A12[0]*B21[1]; A12[1]*B21[0], A12[1]*B21[1]]
    #   This can be done with rank 4 (one term per entry) or less.
    #   Actually, since it's an outer product of a column and a row, it has rank 1 in the matrix sense,
    #   but in tensor decomposition we need bilinear forms.
    #   Let's just use schoolbook for these: 4 terms.
    #
    # - A11*B12: 2x2 * 2x1 = 2x1 (rank 4 schoolbook)
    # - A12*B22: 2x1 * 1x1 = 2x1 (rank 2 schoolbook)
    # - A21*B11: 1x2 * 2x2 = 1x2 (rank 4 schoolbook)
    # - A21*B12: 1x2 * 2x1 = 1x1 (rank 2 schoolbook)
    # - A22*B21: 1x1 * 1x2 = 1x2 (rank 2 schoolbook)
    # - A22*B22: 1x1 * 1x1 = 1x1 (rank 1 schoolbook)
    #
    # Total additional rank: 4 + 4 + 2 + 4 + 2 + 2 + 1 = 19
    # Total rank: 7 + 19 = 26. Correct.
    #
    # Let's add these terms.

    col = 7

    # A12*B21: A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A12 is column 2 of A (rows 0,1), B21 is row 2 of B (cols 0,1)
    # C[i,k] += A[i,2] * B[2,k] for i in 0,1 and k in 0,1
    for i in range(2):
        for k in range(2):
            U[i * 3 + 2, col] = 1.0  # A[i, 2]
            V[2 * 3 + k, col] = 1.0  # B[2, k]
            W[k * 3 + i, col] = 1.0  # C[i, k]
            col += 1

    # A11*B12: A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    # But wait, the Strassen block already handles A11*B11. We need A11*B12 separately.
    # A11 is 2x2 (rows 0,1; cols 0,1), B12 is column 2 of B (rows 0,1)
    # C[i,2] += sum_j A[i,j] * B[j,2] for j in 0,1
    for i in range(2):
        for j in range(2):
            U[i * 3 + j, col] = 1.0  # A[i, j]
            V[j * 3 + 2, col] = 1.0  # B[j, 2]
            W[2 * 3 + i, col] = 1.0  # C[i, 2]
            col += 1

    # A12*B22: A[0:2, 2] * B[2, 2] -> C[0:2, 2]
    for i in range(2):
        U[i * 3 + 2, col] = 1.0  # A[i, 2]
        V[2 * 3 + 2, col] = 1.0  # B[2, 2]
        W[2 * 3 + i, col] = 1.0  # C[i, 2]
        col += 1

    # A21*B11: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    for j in range(2):
        for k in range(2):
            U[2 * 3 + j, col] = 1.0  # A[2, j]
            V[j * 3 + k, col] = 1.0  # B[j, k]
            W[k * 3 + 2, col] = 1.0  # C[2, k]
            col += 1

    # A21*B12: A[2, 0:2] * B[0:2, 2] -> C[2, 2]
    for j in range(2):
        U[2 * 3 + j, col] = 1.0  # A[2, j]
        V[j * 3 + 2, col] = 1.0  # B[j, 2]
        W[2 * 3 + 2, col] = 1.0  # C[2, 2]
        col += 1

    # A22*B21: A[2, 2] * B[2, 0:2] -> C[2, 0:2]
    for k in range(2):
        U[2 * 3 + 2, col] = 1.0  # A[2, 2]
        V[2 * 3 + k, col] = 1.0  # B[2, k]
        W[k * 3 + 2, col] = 1.0  # C[2, k]
        col += 1

    # A22*B22: A[2, 2] * B[2, 2] -> C[2, 2]
    U[2 * 3 + 2, col] = 1.0  # A[2, 2]
    V[2 * 3 + 2, col] = 1.0  # B[2, 2]
    W[2 * 3 + 2, col] = 1.0  # C[2, 2]
    col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    return U, V, W


def _try_solve_for_W(U, V, rank):
    """Given U (nm x rank) and V (mp x rank), solve linear system for W (pn x rank)."""
    n, m, p = 3, 3, 3
    num_vars = (p * n) * rank
    A = np.zeros((n * m * p, num_vars))
    b = np.ones(n * m * p)
    eq = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    var_idx = (k * n + i) * rank + r
                    A[eq, var_idx] = U[i * m + j, r] * V[j * p + k, r]
                eq += 1
    W_flat, residuals, rk, sv = np.linalg.lstsq(A, b, rcond=None)
    W = W_flat.reshape((p * n, rank))
    return W


def _try_rank_fixed_UV(rank, seed=0):
    """Fix U,V as first `rank` schoolbook columns, solve for W, verify."""
    n, m, p = 3, 3, 3
    U_s, V_s, W_s = _schoolbook()
    if rank > U_s.shape[1]:
        return None
    U = U_s[:, :rank].copy()
    V = V_s[:, :rank].copy()
    W = _try_solve_for_W(U, V, rank)
    if _verify(U, V, W):
        return U, V, W
    return None


def _general_search(rank, seed=0, max_attempts=5):
    """
    General search allowing U,V,W to vary. Uses scipy least-squares on a nonlinear system.
    """
    n, m, p = 3, 3, 3
    try:
        from scipy.optimize import least_squares
    except Exception:
        return None

    rng = np.random.default_rng(seed)
    nm, mp, pn = n * m, m * p, p * n

    def residuals(flat):
        U = flat[:nm * rank].reshape(nm, rank)
        V = flat[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W = flat[nm * rank + mp * rank:].reshape(pn, rank)
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    res[idx] = s - 1.0
                    idx += 1
        return res

    for attempt in range(max_attempts):
        if _time_left() < 8.0:
            break
        U0 = rng.normal(size=(nm, rank))
        V0 = rng.normal(size=(mp, rank))
        W0 = rng.normal(size=(pn, rank))
        scale = 1.0 / (rank ** 0.5)
        U0 *= scale
        V0 *= scale
        W0 *= scale
        flat0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
        try:
            sol = least_squares(residuals, flat0, method='lm', max_nfev=2000)
        except Exception:
            continue
        U = sol.x[:nm * rank].reshape(nm, rank)
        V = sol.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W = sol.x[nm * rank + mp * rank:].reshape(pn, rank)
        if _verify(U, V, W, tol=1e-6):
            return U, V, W
    return None


def construct():
    """
    Find a rank-minimal exact decomposition for 3x3 matrix multiplication.
    Tries decreasing ranks; returns the best valid one found before deadline.
    """
    n, m, p = 3, 3, 3

    # Fallback: schoolbook rank 27 (always valid)
    U_s, V_s, W_s = _schoolbook()
    best = (U_s, V_s, W_s)  # rank 27

    # Try rank 26 via Strassen-based construction
    if _time_left() > 15.0:
        U26, V26, W26 = _rank26_decomposition()
        if _verify(U26, V26, W26):
            best = (U26, V26, W26)

    # Try rank 23 (known achievable) via fixed-U,V solve
    for rank in [23, 22, 21, 20, 19]:
        if _time_left() < 15.0:
            break
        res = _try_rank_fixed_UV(rank)
        if res is not None:
            U, V, W = res
            if rank < best[0].shape[1]:
                best = (U, V, W)
            continue
        if _time_left() < 20.0:
            break
        res2 = _general_search(rank)
        if res2 is not None:
            U, V, W = res2
            if rank < best[0].shape[1]:
                best = (U, V, W)
            continue

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


# EVOLVE-BLOCK-END
