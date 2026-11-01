"""Matrix multiplication 2,2,5 - rank-minimal exact tensor decomposition.
Strategy: numerical ALS search seeded from a valid rank-18 structure, then
rationalize and verify. Different design family: optimization-based rather
than purely constructive block composition.
"""

import time
import random
from fractions import Fraction

import numpy as np

DIMENSIONS = [2, 2, 5]


def _verify(U, V, W, n, m, p, rank):
    """Verify exact decomposition using Fraction arithmetic."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if s != 1:
                    return False
    return True


def _build_strassen_2x2():
    """Strassen rank-7 for 2x2*2x2."""
    U = [
        [1, 0, 1, 1, 1, 0, 0],
        [0, 0, -1, 0, 0, 1, -1],
        [0, 1, 0, 0, 1, 0, -1],
        [1, 1, 0, 1, 0, -1, 0],
    ]
    V = [
        [1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
    ]
    W = [
        [1, 0, 0, 1, -1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 0, 0, 0, 1, 1],
    ]
    return U, V, W


def _build_rank18():
    """Rank-18: Strassen + Strassen + schoolbook for last column."""
    n, m, p = 2, 2, 5
    total_rank = 18
    n_rows = n * m
    m_rows = m * p
    p_rows = p * n

    U_full = [[0] * total_rank for _ in range(n_rows)]
    V_full = [[0] * total_rank for _ in range(m_rows)]
    W_full = [[0] * total_rank for _ in range(p_rows)]

    U_s, V_s, W_s = _build_strassen_2x2()

    # Block 1: columns 0,1
    offset = 0
    for i in range(4):
        for c in range(7):
            U_full[i][offset + c] = U_s[i][c]
    v_map = [0, 1, 5, 6]
    for i in range(4):
        for c in range(7):
            V_full[v_map[i]][offset + c] = V_s[i][c]
    w_map = [0, 1, 2, 3]
    for i in range(4):
        for c in range(7):
            W_full[w_map[i]][offset + c] = W_s[i][c]
    offset += 7

    # Block 2: columns 2,3
    for i in range(4):
        for c in range(7):
            U_full[i][offset + c] = U_s[i][c]
    v_map2 = [2, 3, 7, 8]
    for i in range(4):
        for c in range(7):
            V_full[v_map2[i]][offset + c] = V_s[i][c]
    w_map2 = [4, 5, 6, 7]
    for i in range(4):
        for c in range(7):
            W_full[w_map2[i]][offset + c] = W_s[i][c]
    offset += 7

    # Block 3: column 4 schoolbook
    U_full[0][offset + 0] = 1
    U_full[1][offset + 1] = 1
    U_full[2][offset + 2] = 1
    U_full[3][offset + 3] = 1
    V_full[4][offset + 0] = 1
    V_full[9][offset + 1] = 1
    V_full[4][offset + 2] = 1
    V_full[9][offset + 3] = 1
    W_full[8][offset + 0] = 1
    W_full[8][offset + 1] = 1
    W_full[9][offset + 2] = 1
    W_full[9][offset + 3] = 1

    return U_full, V_full, W_full, total_rank


def _build_schoolbook():
    """Full schoolbook rank-20."""
    n, m, p = 2, 2, 5
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, r


def _build_target_tensor(n, m, p):
    """Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _als_decompose(T, rank, n_iters=80, tol=1e-12, seed=42):
    """
    Alternating Least Squares for tensor decomposition.
    T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    """
    rng = np.random.RandomState(seed)
    n1, n2, n3 = T.shape

    U = rng.randn(n1, rank) * 0.5
    V = rng.randn(n2, rank) * 0.5
    W = rng.randn(n3, rank) * 0.5

    for it in range(n_iters):
        # Update U: for each c, U[:,c] = (T x2 (V[:,c] * W[:,c])) / (V[:,c]^T V[:,c] * W[:,c]^T W[:,c])
        # More precisely: fix V,W, solve for U.
        # Residual approach: T - sum_{c' != c} U[:,c']*V[:,c']*W[:,c']
        # Simpler: vectorize.
        # T reshaped: T_flat[i, j*n3+k] = sum_c U[i,c] * V[j,c] * W[k,c]
        # For fixed V,W: T_flat[i,:] = U[i,:] @ (V * W)  where (V*W)[c, j*n3+k] = V[j,c]*W[k,c]

        VW = np.outer(V.ravel(), W.ravel())  # (rank, n2*n3)  -> VW[c, j*n3+k] = V[j,c]*W[k,c]
        # Actually: VW[c, idx] where idx = j*n3+k -> V[j,c]*W[k,c]
        VW = np.zeros((rank, n2 * n3))
        for c in range(rank):
            VW[c, :] = (V[:, c][:, None] * W[:, c][None, :]).ravel()
        T_flat = T.reshape(n1, n2 * n3)
        U_new = np.linalg.lstsq(VW, T_flat, rcond=None)[0].T  # (rank, n1) -> transpose to (n1, rank)
        U_new = U_new.T  # (n1, rank)

        # Update V
        UW = np.zeros((rank, n1 * n3))
        for c in range(rank):
            UW[c, :] = (U[:, c][:, None] * W[:, c][None, :]).ravel()
        T_flat2 = T.reshape(n2, n1 * n3)
        V_new = np.linalg.lstsq(UW, T_flat2, rcond=None)[0].T

        # Update W
        UV = np.zeros((rank, n1 * n2))
        for c in range(rank):
            UV[c, :] = (U[:, c][:, None] * V[:, c][None, :]).ravel()
        T_flat3 = T.reshape(n3, n1 * n2)
        W_new = np.linalg.lstsq(UV, T_flat3, rcond=None)[0].T

        # Check convergence
        err = np.linalg.norm(T - np.einsum('ic,jc,kc->ijk', U_new, V_new, W_new)) / max(np.linalg.norm(T), 1e-15)
        U, V, W = U_new, V_new, W_new
        if err < tol:
            break

    return U, V, W, err


def _rationalize_matrix(M, max_den=12):
    """Round each entry to nearest fraction with denominator <= max_den."""
    result = []
    for row in M:
        new_row = []
        for val in row:
            best_frac = Fraction(0)
            best_err = abs(val)
            for d in range(1, max_den + 1):
                n = round(val * d)
                frac = Fraction(n, d)
                err = abs(val - float(frac))
                if err < best_err:
                    best_err = err
                    best_frac = frac
            new_row.append(best_frac)
        result.append(new_row)
    return result


def _try_als_rank(target_rank, T, deadline, seed=42):
    """Try ALS at given rank, rationalize, verify."""
    U, V, W, err = _als_decompose(T, target_rank, n_iters=60, tol=1e-10, seed=seed)
    if time.time() > deadline:
        return None
    # Rationalize
    U_r = _rationalize_matrix(U.tolist(), max_den=8)
    V_r = _rationalize_matrix(V.tolist(), max_den=8)
    W_r = _rationalize_matrix(W.tolist(), max_den=8)
    if time.time() > deadline:
        return None
    # Verify
    n, m, p = DIMENSIONS
    if _verify(U_r, V_r, W_r, n, m, p, target_rank):
        return U_r, V_r, W_r, target_rank
    return None


def _try_als_from_seed_rank18(target_rank, T, deadline):
    """
    Seed ALS from the rank-18 structure by projecting to lower rank.
    Take the rank-18 factors, randomly project the rank dimension down.
    """
    n, m, p = DIMENSIONS
    U18, V18, W18, _ = _build_rank18()
    U18 = np.array(U18, dtype=float)
    V18 = np.array(V18, dtype=float)
    W18 = np.array(W18, dtype=float)

    rng = np.random.RandomState(123)
    # Random projection from rank 18 to target_rank
    P = rng.randn(18, target_rank)
    U_seed = U18 @ P  # (4, target_rank)
    V_seed = V18 @ P  # (10, target_rank)
    W_seed = W18 @ P  # (10, target_rank)

    # Now run ALS from this seed
    n1, n2, n3 = T.shape
    U, V, W = U_seed, V_seed, W_seed

    for it in range(50):
        if time.time() > deadline:
            return None
        VW = np.zeros((target_rank, n2 * n3))
        for c in range(target_rank):
            VW[c, :] = (V[:, c][:, None] * W[:, c][None, :]).ravel()
        T_flat = T.reshape(n1, n2 * n3)
        U = np.linalg.lstsq(VW, T_flat, rcond=None)[0].T

        UW = np.zeros((target_rank, n1 * n3))
        for c in range(target_rank):
            UW[c, :] = (U[:, c][:, None] * W[:, c][None, :]).ravel()
        T_flat2 = T.reshape(n2, n1 * n3)
        V = np.linalg.lstsq(UW, T_flat2, rcond=None)[0].T

        UV = np.zeros((target_rank, n1 * n2))
        for c in range(target_rank):
            UV[c, :] = (U[:, c][:, None] * V[:, c][None, :]).ravel()
        T_flat3 = T.reshape(n3, n1 * n2)
        W = np.linalg.lstsq(UV, T_flat3, rcond=None)[0].T

    if time.time() > deadline:
        return None

    U_r = _rationalize_matrix(U.tolist(), max_den=8)
    V_r = _rationalize_matrix(V.tolist(), max_den=8)
    W_r = _rationalize_matrix(W.tolist(), max_den=8)
    if _verify(U_r, V_r, W_r, n, m, p, target_rank):
        return U_r, V_r, W_r, target_rank
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)

    # First, get the guaranteed rank-18 solution
    U18, V18, W18, r18 = _build_rank18()
    if _verify(U18, V18, W18, n, m, p, r18):
        best_result = {"U": U18, "V": V18, "W": W18}
        best_rank = r18
    else:
        U_sb, V_sb, W_sb, r_sb = _build_schoolbook()
        best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
        best_rank = r_sb

    # Try ALS for rank 16
    if time.time() < deadline - 5:
        result = _try_als_from_seed_rank18(16, T, deadline - 5)
        if result is not None:
            best_result = {"U": result[0], "V": result[1], "W": result[2]}
            best_rank = result[3]

    # Try random-init ALS for rank 16 with different seeds
    if time.time() < deadline - 5 and best_rank > 16:
        for seed in [7, 13, 99, 256]:
            if time.time() > deadline - 5:
                break
            result = _try_als_rank(16, T, deadline - 5, seed=seed)
            if result is not None:
                best_result = {"U": result[0], "V": result[1], "W": result[2]}
                best_rank = result[3]
                break

    # Try rank 15
    if time.time() < deadline - 5 and best_rank > 15:
        for seed in [3, 42, 77]:
            if time.time() > deadline - 5:
                break
            result = _try_als_rank(15, T, deadline - 5, seed=seed)
            if result is not None:
                best_result = {"U": result[0], "V": result[1], "W": result[2]}
                best_rank = result[3]
                break

    # Try rank 17 as intermediate
    if time.time() < deadline - 5 and best_rank > 17:
        for seed in [1, 5, 11]:
            if time.time() > deadline - 5:
                break
            result = _try_als_rank(17, T, deadline - 5, seed=seed)
            if result is not None:
                best_result = {"U": result[0], "V": result[1], "W": result[2]}
                best_rank = result[3]
                break

    return best_result


# EVOLVE-BLOCK-END
