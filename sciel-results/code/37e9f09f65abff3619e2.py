"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START


def _build_trivial(n, m, p):
    """Return the rank-nmp trivial decomposition as (U, V, W) lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u][r] = 1
        V[idx_v][r] = 1
        W[idx_w][r] = 1
    return U, V, W, rank


def _validate(n, m, p, U, V, W):
    """Verify that sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True


def _build_strassen(n, m, p):
    """
    Build a rank-55 decomposition for 2x3 * 3x10 using Strassen on 2x2x2 blocks.
    Each column pair uses 7 Strassen products + 4 schoolbook products for row 2.
    """
    assert n == 2 and m == 3 and p == 10

    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55

    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1

        # M1: (a00 + a11) * (B[0,k0] + B[1,k1])
        U[0][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        V[10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M2: (a10 + a11) * B[0,k0]
        U[3][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1

        # M3: a00 * (B[0,k1] - B[1,k1])
        U[0][col] += 1
        V[k1][col] += 1
        V[10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M4: a11 * (B[1,k0] - B[0,k0])
        U[4][col] += 1
        V[10 + k0][col] += 1
        V[k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1

        # M5: (a00 + a01) * B[1,k1]
        U[0][col] += 1
        U[1][col] += 1
        V[10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1

        # M6: (a10 - a00) * (B[0,k0] + B[0,k1])
        U[3][col] += 1
        U[0][col] -= 1
        V[k0][col] += 1
        V[k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M7: (a01 - a11) * (B[1,k0] + B[1,k1])
        U[1][col] += 1
        U[4][col] -= 1
        V[10 + k0][col] += 1
        V[10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1

        # Schoolbook for A[:,2] * B[2, k0:k1+1]
        U[2][col] += 1
        V[20 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1

        U[2][col] += 1
        V[20 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1

        U[5][col] += 1
        V[20 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1

        U[5][col] += 1
        V[20 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    return U, V, W, rank


def _try_reduce_rank(n, m, p, U, V, W, deadline):
    """
    Greedily remove columns that are redundant. Uses incremental tensor updates
    for speed. Tries multiple random orderings to escape local optima.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    target = np.ones((n, m, p))

    # Precompute per-column tensor contributions
    col_contrib = np.zeros((rank, n, m, p))
    for r in range(rank):
        u_col = U_arr[:, r].reshape(n, m)
        v_col = V_arr[:, r].reshape(m, p)
        w_col = W_arr[:, r].reshape(p, n)
        col_contrib[r] = np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)

    best_rank = rank
    best_keep = np.ones(rank, dtype=bool)

    # Try multiple random orderings
    rng = np.random.default_rng(42)
    for attempt in range(20):
        if time.time() >= deadline:
            break

        keep = np.ones(rank, dtype=bool)
        T_current = col_contrib.sum(axis=0)

        # Random permutation of column indices
        order = rng.permutation(rank)

        changed = True
        while changed and time.time() < deadline:
            changed = False
            for r in order:
                if not keep[r]:
                    continue
                T_new = T_current - col_contrib[r]
                if np.allclose(T_new, target, atol=1e-9):
                    keep[r] = False
                    T_current = T_new
                    changed = True
                    break

        new_rank = int(np.sum(keep))
        if new_rank < best_rank:
            best_rank = new_rank
            best_keep = keep.copy()

    if best_rank >= rank:
        return U, V, W, rank

    U_new = U_arr[:, best_keep]
    V_new = V_arr[:, best_keep]
    W_new = W_arr[:, best_keep]

    U_list = U_new.tolist()
    V_list = V_new.tolist()
    W_list = W_new.tolist()

    if _validate(n, m, p, U_list, V_list, W_list):
        return U_list, V_list, W_list, best_rank
    else:
        return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin under 150s cap

    n, m, p = DIMENSIONS

    U, V, W, rank = _build_strassen(n, m, p)

    if not _validate(n, m, p, U, V, W):
        U, V, W, rank = _build_trivial(n, m, p)
    else:
        U, V, W, rank = _try_reduce_rank(n, m, p, U, V, W, deadline)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
