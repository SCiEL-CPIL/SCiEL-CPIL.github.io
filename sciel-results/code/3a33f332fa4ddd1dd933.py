"""Matrix multiplication 2x4x5 - rank-minimal exact tensor decomposition.

Strategy:
1. Build a known rank-36 decomposition using Strassen on 2x2 sub-blocks.
2. Attempt to find lower-rank decompositions via:
   a. Structured search over block partitions
   b. ALS optimization with multiple restarts
   c. Border-rank perturbation from rank-36
"""

import time
import numpy as np

DIMENSIONS = [2, 4, 5]
_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0


def _time_remaining():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)


def _build_strassen_decomp():
    """Build the rank-36 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 36
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_strassen(a_row_off, a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        c = start_col
        # M1: (p00+p11)*(q00+q11) -> R00+1, R11+1
        U[0*4 + a_col_off + 0, c] = 1
        U[1*4 + a_col_off + 1, c] = 1
        V[(b_row_off+0)*5 + b_col_off + 0, c] = 1
        V[(b_row_off+1)*5 + b_col_off + 1, c] = 1
        W[(c_col_off+0)*2 + 0, c] = 1
        W[(c_col_off+1)*2 + 1, c] = 1
        c += 1
        # M2: (p10+p11)*(q00+q01) -> R10+1
        U[1*4 + a_col_off + 0, c] = 1
        U[1*4 + a_col_off + 1, c] = 1
        V[(b_row_off+0)*5 + b_col_off + 0, c] = 1
        V[(b_row_off+0)*5 + b_col_off + 1, c] = 1
        W[(c_col_off+0)*2 + 1, c] = 1
        c += 1
        # M3: (p00+p01)*(q10+q11) -> R01+1
        U[0*4 + a_col_off + 0, c] = 1
        U[0*4 + a_col_off + 1, c] = 1
        V[(b_row_off+1)*5 + b_col_off + 0, c] = 1
        V[(b_row_off+1)*5 + b_col_off + 1, c] = 1
        W[(c_col_off+1)*2 + 0, c] = 1
        c += 1
        # M4: p00*(q00-q11) -> R00+1, R11-1
        U[0*4 + a_col_off + 0, c] = 1
        V[(b_row_off+0)*5 + b_col_off + 0, c] = 1
        V[(b_row_off+1)*5 + b_col_off + 1, c] = -1
        W[(c_col_off+0)*2 + 0, c] = 1
        W[(c_col_off+1)*2 + 1, c] = -1
        c += 1
        # M5: p11*(q01-q00) -> R00-1, R01+1
        U[1*4 + a_col_off + 1, c] = 1
        V[(b_row_off+0)*5 + b_col_off + 1, c] = 1
        V[(b_row_off+0)*5 + b_col_off + 0, c] = -1
        W[(c_col_off+0)*2 + 0, c] = -1
        W[(c_col_off+1)*2 + 0, c] = 1
        c += 1
        # M6: (p01+p11)*(q10-q00) -> R11+1
        U[0*4 + a_col_off + 1, c] = 1
        U[1*4 + a_col_off + 1, c] = 1
        V[(b_row_off+1)*5 + b_col_off + 0, c] = 1
        V[(b_row_off+0)*5 + b_col_off + 0, c] = -1
        W[(c_col_off+1)*2 + 1, c] = 1
        c += 1
        # M7: (p00+p10)*(q11-q10) -> R00-1, R10+1
        U[0*4 + a_col_off + 0, c] = 1
        U[1*4 + a_col_off + 0, c] = 1
        V[(b_row_off+1)*5 + b_col_off + 1, c] = 1
        V[(b_row_off+1)*5 + b_col_off + 0, c] = -1
        W[(c_col_off+0)*2 + 0, c] = -1
        W[(c_col_off+0)*2 + 1, c] = 1
        c += 1
        return c

    def add_schoolbook_2x2x1(a_row_off, a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        c = start_col
        U[0*4 + a_col_off + 0, c] = 1
        V[(b_row_off+0)*5 + b_col_off, c] = 1
        W[c_col_off*2 + 0, c] = 1
        c += 1
        U[0*4 + a_col_off + 1, c] = 1
        V[(b_row_off+1)*5 + b_col_off, c] = 1
        W[c_col_off*2 + 0, c] = 1
        c += 1
        U[1*4 + a_col_off + 0, c] = 1
        V[(b_row_off+0)*5 + b_col_off, c] = 1
        W[c_col_off*2 + 1, c] = 1
        c += 1
        U[1*4 + a_col_off + 1, c] = 1
        V[(b_row_off+1)*5 + b_col_off, c] = 1
        W[c_col_off*2 + 1, c] = 1
        c += 1
        return c

    col = add_strassen(0, 0, 0, 0, 0, col)
    col = add_strassen(0, 2, 2, 0, 0, col)
    col = add_strassen(0, 0, 0, 2, 2, col)
    col = add_strassen(0, 2, 2, 2, 2, col)
    col = add_schoolbook_2x2x1(0, 0, 0, 4, 4, col)
    col = add_schoolbook_2x2x1(0, 2, 2, 4, 4, col)

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _build_schoolbook():
    """Build the rank-40 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, tol=1e-9):
    """Verify tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U_idx = U.reshape(n, m, rank)
    V_idx = V.reshape(m, p, rank)
    W_idx = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx)
    return np.allclose(T, np.ones((n, m, p)), atol=tol)


def _verify_numeric(U, V, W, tol=1e-9):
    """Verify by computing matrix products."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(12345)
    A_test = rng.randn(n, m)
    B_test = rng.randn(m, p)
    C_expected = A_test @ B_test

    A_flat = A_test.flatten()
    B_flat = B_test.flatten()
    U_TA = U.T @ A_flat
    V_TB = V.T @ B_flat

    C_tensor = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            C_tensor[i, k] = np.dot(U_TA * V_TB, W[k * n + i, :])

    return np.allclose(C_tensor, C_expected, atol=tol)


def _als_decompose(target_rank, max_iters=2000, tol=1e-10, seed=42):
    """ALS optimization for tensor decomposition of matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)

    U = rng.randn(n * m, target_rank) * 0.1
    V = rng.randn(m * p, target_rank) * 0.1
    W = rng.randn(p * n, target_rank) * 0.1

    def solve_U():
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    x = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except np.linalg.LinAlgError:
                    x = U[i * m + j, :].copy()
                U[i * m + j, :] = x

    def solve_V():
        for j in range(m):
            for k in range(p):
                N = np.zeros((n, target_rank))
                for i in range(n):
                    N[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    x = np.linalg.lstsq(N, np.ones(n), rcond=None)[0]
                except np.linalg.LinAlgError:
                    x = V[j * p + k, :].copy()
                V[j * p + k, :] = x

    def solve_W():
        for k in range(p):
            for i in range(n):
                O = np.zeros((m, target_rank))
                for j in range(m):
                    O[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    x = np.linalg.lstsq(O, np.ones(m), rcond=None)[0]
                except np.linalg.LinAlgError:
                    x = W[k * n + i, :].copy()
                W[k * n + i, :] = x

    prev_error = float('inf')
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break

        solve_U()
        solve_V()
        solve_W()

        U_idx = U.reshape(n, m, target_rank)
        V_idx = V.reshape(m, p, target_rank)
        W_idx = W.reshape(p, n, target_rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx)
        error = np.max(np.abs(T - 1.0))

        if error < tol:
            break
        if abs(error - prev_error) < 1e-15:
            U += rng.randn(*U.shape) * 1e-4
            V += rng.randn(*V.shape) * 1e-4
            W += rng.randn(*W.shape) * 1e-4

        prev_error = error

    return U, V, W, error


def _try_partition_search():
    """
    Try different block partitions of the intermediate dimension m=4.
    Partition into sizes [3,1] and [1,3] and [2,2].
    For each partition, build a structured decomposition and see if we can
    find a lower-rank one.
    """
    n, m, p = DIMENSIONS
    best_rank = 36
    best_U, best_V, best_W = None, None, None

    # Partition [3,1]: A = [A_3x2 | A_1x2], B = [B_3x5; B_1x5]
    # C = A_3x2 @ B_3x5 + A_1x2 @ B_1x5
    # This is 2x3 * 3x5 plus 2x1 * 1x5
    # 2x3*3x5: can use Strassen-like for 2x2 sub-blocks
    # 2x1*1x5: schoolbook, 2 terms

    # Let's try to build a decomposition for 2x3 * 3x5
    # Split 3 into 2+1: A_3x2 = [A2 | A1], B_3x5 = [B2; B1]
    # C = A2@B2 + A1@B1
    # A2 is 2x2, B2 is 2x5, A1 is 2x1, B1 is 1x5
    # A2@B2: 2x2 * 2x5 -> split B2 into [B2a | B2b] where B2a is 2x2, B2b is 2x3
    # A2@B2a: 2x2*2x2 -> Strassen, 7 terms
    # A2@B2b: 2x2*2x3 -> schoolbook, 6 terms (or try to optimize)
    # A1@B1: 2x1*1x5 -> 2 terms
    # Total for 2x3*3x5: 7 + 6 + 2 = 15 terms
    # Plus 2x1*1x5: 2 terms
    # Total: 17 terms

    # Let's try to build this
    rank_candidate = 17
    U = np.zeros((n * m, rank_candidate), dtype=float)
    V = np.zeros((m * p, rank_candidate), dtype=float)
    W = np.zeros((p * n, rank_candidate), dtype=float)
    col = 0

    # A2@B2a: 2x2 * 2x2 at A[0:2,0:2], B[0:2,0:2], C[0:2,0:2]
    # Strassen
    # M1: (p00+p11)*(q00+q11)
    U[0, col] = 1; U[5, col] = 1
    V[0, col] = 1; V[6, col] = 1
    W[0, col] = 1; W[3, col] = 1
    col += 1
    # M2: (p10+p11)*(q00+q01)
    U[4, col] = 1; U[5, col] = 1
    V[0, col] = 1; V[1, col] = 1
    W[2, col] = 1
    col += 1
    # M3: (p00+p01)*(q10+q11)
    U[0, col] = 1; U[1, col] = 1
    V[5, col] = 1; V[6, col] = 1
    W[1, col] = 1
    col += 1
    # M4: p00*(q00-q11)
    U[0, col] = 1
    V[0, col] = 1; V[6, col] = -1
    W[0, col] = 1; W[3, col] = -1
    col += 1
    # M5: p11*(q01-q00)
    U[5, col] = 1
    V[1, col] = 1; V[0, col] = -1
    W[0, col] = -1; W[1, col] = 1
    col += 1
    # M6: (p01+p11)*(q10-q00)
    U[1, col] = 1; U[5, col] = 1
    V[5, col] = 1; V[0, col] = -1
    W[3, col] = 1
    col += 1
    # M7: (p00+p10)*(q11-q10)
    U[0, col] = 1; U[4, col] = 1
    V[6, col] = 1; V[5, col] = -1
    W[0, col] = -1; W[2, col] = 1
    col += 1

    # A2@B2b: 2x2 * 2x3 at A[0:2,0:2], B[0:2,2:5], C[0:2,2:5]
    # Schoolbook: 6 terms
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i * 4 + j, col] = 1
                V[j * 5 + 2 + k, col] = 1
                W[(2 + k) * 2 + i, col] = 1
                col += 1

    # A1@B1: 2x1 * 1x5 at A[0:2,2:3], B[2:3,0:5], C[0:2,0:5]
    for i in range(2):
        for k in range(5):
            U[i * 4 + 2, col] = 1
            V[2 * 5 + k, col] = 1
            W[k * 2 + i, col] = 1
            col += 1

    # Now we have used col terms. Check if col == rank_candidate
    # If not, adjust
    if col != rank_candidate:
        # Pad or truncate
        if col < rank_candidate:
            U = U[:, :col]
            V = V[:, :col]
            W = W[:, :col]
            rank_candidate = col
        else:
            U = U[:, :rank_candidate]
            V = V[:, :rank_candidate]
            W = W[:, :rank_candidate]

    # Verify
    if _verify(U, V, W, tol=1e-9) and _verify_numeric(U, V, W, tol=1e-9):
        best_rank = rank_candidate
        best_U, best_V, best_W = U, V, W

    return best_rank, best_U, best_V, best_W


def construct():
    """
    Find the lowest-rank exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS

    # Start with known rank-36 decomposition
    U36, V36, W36 = _build_strassen_decomp()

    if not _verify(U36, V36, W36):
        U40, V40, W40 = _build_schoolbook()
        return {"U": U40.tolist(), "V": V40.tolist(), "W": W40.tolist()}

    best_rank = 36
    best_U, best_V, best_W = U36, V36, W36

    # Try partition search
    if _time_remaining() > 30.0:
        pr, pU, pV, pW = _try_partition_search()
        if pr < best_rank and pU is not None:
            best_rank = pr
            best_U, best_V, best_W = pU, pV, pW

    # Try ALS at various ranks
    for target_rank in range(24, best_rank):
        if _time_remaining() < 10.0:
            break

        for seed in range(8):
            if _time_remaining() < 5.0:
                break

            U_try, V_try, W_try, error = _als_decompose(
                target_rank, max_iters=800, tol=1e-10, seed=seed
            )

            if _time_remaining() < 3.0:
                break

            if error < 1e-8:
                if _verify(U_try, V_try, W_try, tol=1e-8) and _verify_numeric(U_try, V_try, W_try, tol=1e-8):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_try, V_try, W_try
                    break

    # Border-rank perturbation from rank-36
    if _time_remaining() > 15.0 and best_rank == 36:
        for col_to_remove in range(36):
            if _time_remaining() < 8.0:
                break

            mask = [i for i in range(36) if i != col_to_remove]
            U_reduced = U36[:, mask]
            V_reduced = V36[:, mask]
            W_reduced = W36[:, mask]

            eps = 1e-3
            rng = np.random.RandomState(col_to_remove)
            U_init = U_reduced + rng.randn(*U_reduced.shape) * eps
            V_init = V_reduced + rng.randn(*V_reduced.shape) * eps
            W_init = W_reduced + rng.randn(*W_reduced.shape) * eps

            target_rank = 35
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()

            for it in range(300):
                if _time_remaining() < 5.0:
                    break

                # Solve U
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            M[k, :] = V_cur[j * p + k, :] * W_cur[k * n + i, :]
                        try:
                            x = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                        except np.linalg.LinAlgError:
                            x = U_cur[i * m + j, :].copy()
                        U_cur[i * m + j, :] = x

                # Solve V
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, target_rank))
                        for i in range(n):
                            N[i, :] = U_cur[i * m + j, :] * W_cur[k * n + i, :]
                        try:
                            x = np.linalg.lstsq(N, np.ones(n), rcond=None)[0]
                        except np.linalg.LinAlgError:
                            x = V_cur[j * p + k, :].copy()
                        V_cur[j * p + k, :] = x

                # Solve W
                for k in range(p):
                    for i in range(n):
                        O = np.zeros((m, target_rank))
                        for j in range(m):
                            O[j, :] = U_cur[i * m + j, :] * V_cur[j * p + k, :]
                        try:
                            x = np.linalg.lstsq(O, np.ones(m), rcond=None)[0]
                        except np.linalg.LinAlgError:
                            x = W_cur[k * n + i, :].copy()
                        W_cur[k * n + i, :] = x

                U_idx = U_cur.reshape(n, m, target_rank)
                V_idx = V_cur.reshape(m, p, target_rank)
                W_idx = W_cur.reshape(p, n, target_rank)
                T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx)
                error = np.max(np.abs(T - 1.0))

                if error < 1e-8:
                    if _verify(U_cur, V_cur, W_cur, tol=1e-8) and _verify_numeric(U_cur, V_cur, W_cur, tol=1e-8):
                        best_rank = 35
                        best_U, best_V, best_W = U_cur, V_cur, W_cur
                    break

    # Final verification
    if not _verify(best_U, best_V, best_W, tol=1e-8):
        best_U, best_V, best_W = U36, V36, W36
        best_rank = 36

    if not _verify(best_U, best_V, best_W, tol=1e-8):
        U40, V40, W40 = _build_schoolbook()
        best_U, best_V, best_W = U40, V40, W40
        best_rank = 40

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"U shape: {len(result['U'])}x{len(result['U'][0])}")
    print(f"V shape: {len(result['V'])}x{len(result['V'][0])}")
    print(f"W shape: {len(result['W'])}x{len(result['W'][0])}")
