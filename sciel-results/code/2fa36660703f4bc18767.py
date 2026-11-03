"""Slice-method decomposition for 3x4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]


def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c]
        Vc = V[:, c]
        Wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Uc[i * m + j] * Vc[j * p + k] * Wc[k * n + i]
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)


def _search_decomposition(n, m, p, max_rank, coeff_set, timeout=50):
    """Search for rank-minimal decomposition of n x m x p tensor."""
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n

    for r in range(1, max_rank + 1):
        if time.time() - start_time > timeout:
            break

        num_trials = 8000
        for trial in range(num_trials):
            if time.time() - start_time > timeout:
                break

            # Random U, V with small integer/half-integer entries
            U_rand = np.random.choice(coeff_set, size=(nm, r))
            V_rand = np.random.choice(coeff_set, size=(mp, r))

            # Set up linear system for W
            # Variables: W[k*n+i, c] for k in range(p), i in range(n), c in range(r)
            # Total variables: p * n * r
            # Equation for (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1

            A = np.zeros((n * m * p, p * n * r))
            b = np.ones(n * m * p)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            col = (k * n + i) * r + c
                            A[eq_idx, col] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
                        eq_idx += 1

            try:
                W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                if np.allclose(A @ W_flat, b, atol=1e-8):
                    W_rand = W_flat.reshape((p * n, r))
                    if _check_decomposition(U_rand, V_rand, W_rand, n, m, p):
                        # Check entries are "nice" (integer or half-integer)
                        W_rounded = np.round(W_rand * 2) / 2
                        if np.allclose(W_rounded, W_rand, atol=1e-8):
                            return (r, U_rand.tolist(), V_rand.tolist(), W_rounded.tolist())
            except:
                pass

    return None


def _construct_sub(n, m, p, max_rank, timeout=45):
    """Construct decomposition for n x m x p."""
    coeff_set = [-2, -1, -0.5, 0, 0.5, 1, 2]
    result = _search_decomposition(n, m, p, max_rank, coeff_set, timeout)
    if result:
        return result
    # Fallback: schoolbook
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
    return (rank, U, V, W)


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7

    # Partition the 7 output columns (k dimension) into blocks: 3, 2, 2
    blocks = [3, 2, 2]
    k_offsets = []
    offset = 0
    for b in blocks:
        k_offsets.append(offset)
        offset += b

    # Find sub-decompositions
    r0, U0, V0, W0 = _construct_sub(n, m, blocks[0], max_rank=20, timeout=35)
    r1, U1, V1, W1 = _construct_sub(n, m, blocks[1], max_rank=16, timeout=30)
    r2, U2, V2, W2 = _construct_sub(n, m, blocks[2], max_rank=16, timeout=30)

    total_rank = r0 + r1 + r2

    # Combine: U is (n*m, total_rank), V is (m*p, total_rank), W is (p*n, total_rank)
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]

    sub_decomps = [
        (r0, U0, V0, W0, k_offsets[0], blocks[0]),
        (r1, U1, V1, W1, k_offsets[1], blocks[1]),
        (r2, U2, V2, W2, k_offsets[2], blocks[2]),
    ]

    col_offset = 0
    for r, U_sub, V_sub, W_sub, k_off, b in sub_decomps:
        # U: same input indices across blocks
        for i in range(n * m):
            for c in range(r):
                U_full[i][col_offset + c] = U_sub[i][c]

        # V: V_sub has shape (m*b, r) with indices (j, k_local)
        for j in range(m):
            for kk in range(b):
                k_global = k_off + kk
                row_v = j * p + k_global
                for c in range(r):
                    V_full[row_v][col_offset + c] = V_sub[j * b + kk][c]

        # W: W_sub has shape (b*n, r) with indices (k_local, i)
        for kk in range(b):
            k_global = k_off + kk
            for i in range(n):
                row_w = k_global * n + i
                for c in range(r):
                    W_full[row_w][col_offset + c] = W_sub[kk * n + i][c]

        col_offset += r

    return {"U": U_full, "V": V_full, "W": W_full}
