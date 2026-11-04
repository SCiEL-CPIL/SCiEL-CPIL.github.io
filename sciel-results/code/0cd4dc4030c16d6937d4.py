"""
Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Strategy:
  - Build rank-77 Strassen decomposition as baseline.
  - For target ranks 70..76, use Gauss-Newton / Levenberg-Marquardt
    to find exact decompositions by minimizing the tensor residual.
  - Return the lowest-rank valid decomposition found.
"""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [4, 4, 6]

_STRassen_U = np.array([
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1],
], dtype=np.float64)

_STRassen_V = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
], dtype=np.float64)

_STRassen_W = np.array([
    [1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 1, 0],
], dtype=np.float64)


def _build_rank77(n, m, p):
    """Build rank-77 Strassen-based decomposition for n x m * m x p."""
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)

    outer_U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1},
    ]
    outer_V_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    outer_W_coeffs = [
        {0: 1, 3: 1},
        {1: 1},
        {0: -1},
        {1: 1},
        {3: 1},
        {2: 1},
        {0: 1, 2: -1},
    ]

    # Part 1: 4x4 * 4x4 -> 4x4 (columns 0..48)
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            for block_idx, coeff in outer_U_coeffs[s].items():
                block_offset = block_idx * 4
                for local_idx in range(4):
                    inner_u = _STRassen_U[local_idx][t]
                    if inner_u != 0:
                        U[block_offset + local_idx][col] += coeff * inner_u
            for block_idx, coeff in outer_V_coeffs[s].items():
                block_offset = block_idx * 4
                for local_idx in range(4):
                    inner_v = _STRassen_V[local_idx][t]
                    if inner_v != 0:
                        V[block_offset + local_idx][col] += coeff * inner_v
            for block_idx, coeff in outer_W_coeffs[s].items():
                if block_idx == 0:
                    w_offsets = [0, 1, 4, 5]
                elif block_idx == 1:
                    w_offsets = [2, 3, 6, 7]
                elif block_idx == 2:
                    w_offsets = [8, 9, 12, 13]
                else:
                    w_offsets = [10, 11, 14, 15]
                for local_idx, w_off in enumerate(w_offsets):
                    inner_w = _STRassen_W[local_idx][t]
                    if inner_w != 0:
                        W[w_off][col] += coeff * inner_w

    # Part 2: 4x4 * 4x2 -> 4x2 (columns 49..76)
    blocks_442 = [
        (0, 'top', 'top'),
        (1, 'bot', 'top'),
        (2, 'top', 'bot'),
        (3, 'bot', 'bot'),
    ]
    for bi, (a_block, b_part, c_part) in enumerate(blocks_442):
        for t in range(7):
            col = rank_444 + bi * 7 + t
            a_offset = a_block * 4
            for local_idx in range(4):
                inner_u = _STRassen_U[local_idx][t]
                if inner_u != 0:
                    U[a_offset + local_idx][col] = inner_u
            if b_part == 'top':
                v_offsets = [4, 5, 10, 11]
            else:
                v_offsets = [16, 17, 22, 23]
            for local_idx, v_off in enumerate(v_offsets):
                inner_v = _STRassen_V[local_idx][t]
                if inner_v != 0:
                    V[v_off][col] = inner_v
            if c_part == 'top':
                w_offsets = [16, 17, 20, 21]
            else:
                w_offsets = [18, 19, 22, 23]
            for local_idx, w_off in enumerate(w_offsets):
                inner_w = _STRassen_W[local_idx][t]
                if inner_w != 0:
                    W[w_off][col] = inner_w

    return U, V, W


def _build_target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _tensor_residual(params, T, n, m, p, rank):
    """Compute residual of tensor decomposition from flattened params."""
    nm, mp, pn = n * m, m * p, p * n
    U = params[:nm * rank].reshape(nm, rank)
    V = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
    W = params[nm * rank + mp * rank:].reshape(pn, rank)
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return (T - recon).ravel()


def _try_reduce_rank(U0, V0, W0, T, n, m, p, target_rank, max_iter=200, deadline=None):
    """Try to find a rank-target_rank decomposition using least_squares."""
    nm, mp, pn = n * m, m * p, p * n
    # Initialize: take first target_rank columns
    U_init = U0[:, :target_rank].copy()
    V_init = V0[:, :target_rank].copy()
    W_init = W0[:, :target_rank].copy()
    x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])

    def residual(x):
        return _tensor_residual(x, T, n, m, p, target_rank)

    try:
        result = least_squares(
            residual, x0, method='lm', max_nfev=max_iter,
            xtol=1e-14, ftol=1e-14, gtol=1e-14
        )
    except Exception:
        return None

    if deadline is not None and time.time() > deadline:
        return None

    final_res = result.fun
    err = np.sum(final_res ** 2)
    if err < 1e-10:
        x = result.x
        U = x[:nm * target_rank].reshape(nm, target_rank)
        V = x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
        W = x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)
        # Verify
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(recon, T, atol=1e-9):
            return U, V, W
    return None


def construct():
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s

    n, m, p = DIMENSIONS  # 4, 4, 6
    T = _build_target_tensor(n, m, p)

    # Build rank-77 baseline
    U77, V77, W77 = _build_rank77(n, m, p)

    # Verify baseline
    recon = np.einsum('ar,br,cr->abc', U77, V77, W77)
    if not np.allclose(recon, T, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U_s = np.zeros((n * m, rank), dtype=np.float64)
        V_s = np.zeros((m * p, rank), dtype=np.float64)
        W_s = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    best_U, best_V, best_W = U77, V77, W77
    best_rank = 77

    # Try to reduce rank from 76 down to 65
    for target_rank in range(76, 64, -1):
        if time.time() > deadline - 15:
            break

        # Try multiple random initializations
        found = False
        for attempt in range(3):
            if time.time() > deadline - 15:
                break

            if attempt == 0:
                # Use first target_rank columns
                U_init = U77[:, :target_rank].copy()
                V_init = V77[:, :target_rank].copy()
                W_init = W77[:, :target_rank].copy()
            else:
                # Random perturbation of the full 77-col solution, then truncate
                rng = np.random.RandomState(42 + attempt)
                U_init = U77[:, :target_rank] + rng.randn(n * m, target_rank) * 0.01
                V_init = V77[:, :target_rank] + rng.randn(m * p, target_rank) * 0.01
                W_init = W77[:, :target_rank] + rng.randn(p * n, target_rank) * 0.01

            result = _try_reduce_rank(U_init, V_init, W_init, T, n, m, p, target_rank,
                                       max_iter=100, deadline=deadline - 15)
            if result is not None:
                U_r, V_r, W_r = result
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                found = True
                break

    # Final verification
    recon = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
    if np.allclose(recon, T, atol=1e-9):
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to rank-77
        return {
            "U": U77.tolist(),
            "V": V77.tolist(),
            "W": W77.tolist()
        }
