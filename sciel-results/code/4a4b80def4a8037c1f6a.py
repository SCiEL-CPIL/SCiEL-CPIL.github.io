# EVOLVE-BLOCK-START
import time
import numpy as np


def _swept_1d(x, N, c):
    """Boolean array of length len(c): whether sample c is inside tube k's sweep.
    x: (N,) base positions; k: tube index 0..N-1."""
    k = np.arange(N, dtype=np.float64)
    # (N, M)
    L = x[:, None] + c[None, :] * k[:, None] / N
    U = L + 1.0 / N
    return (c[None, :] >= L) & (c[None, :] < U)


def _union_1d_len(x, N, c):
    """Estimate 1D union length via fixed sample c in [0,1]."""
    hit = _swept_1d(x, N, c).any(axis=0)
    return hit.mean()


def _eval_1d(x, N, c):
    """Return (union_len, box_len)."""
    k = np.arange(N, dtype=np.float64)
    box_len = np.max(x + 1.0 / N) - np.min(x)
    hit = _swept_1d(x, N, c).any(axis=0)
    return hit.mean(), box_len


def _local_search_1d(x, N, c_eval, c_check, max_iter, step0, step_decay, patience):
    """Gradient-free local search minimizing union_len * box_len."""
    cur = np.array(x, dtype=np.float64)
    cur_len, cur_box = _eval_1d(cur, N, c_eval)
    cur_score = cur_len * cur_box
    if not np.isfinite(cur_score) or cur_box <= 1e-9:
        cur_score = np.inf
    step = step0
    best_x = cur.copy()
    best_score = cur_score
    stall = 0
    for it in range(max_iter):
        improved = False
        for k in range(N):
            for d in (1, -1):
                cand = cur.copy()
                cand[k] += d * step
                l, b = _eval_1d(cand, N, c_eval)
                s = l * b
                if np.isfinite(s) and b > 1e-9 and s < cur_score - 1e-12:
                    cur = cand
                    cur_score = s
                    improved = True
                    if s < best_score:
                        best_score = s
                        best_x = cur.copy()
        if not improved:
            step *= step_decay
            stall += 1
            if stall >= patience:
                break
        else:
            stall = 0
    # verify best on check sample
    bl, bb = _eval_1d(best_x, N, c_check)
    if np.isfinite(bl * bb) and bb > 1e-9 and bl * bb < best_score:
        best_score = bl * bb
    return best_x, best_score


def _initial_keich(N):
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 620.0  # safety margin before 630s kill

    # Fixed evaluation samples (deterministic)
    rng = np.random.default_rng(7)
    c_eval = rng.uniform(0, 1, 400000)
    rng2 = np.random.default_rng(99)
    c_check = rng2.uniform(0, 1, 400000)

    base = _initial_keich(N)

    # Local search in 1D for x and y (identical problem)
    x_best, x_score = _local_search_1d(
        base, N, c_eval, c_check,
        max_iter=200, step0=0.05, step_decay=0.6, patience=30,
    )
    if time.time() > deadline:
        y_best = x_best
    else:
        # Use a perturbed start for y to explore different basin
        rng3 = np.random.default_rng(5)
        y_start = base + rng3.uniform(-0.02, 0.02, N)
        y_best, y_score = _local_search_1d(
            y_start, N, c_eval, c_check,
            max_iter=200, step0=0.05, step_decay=0.6, patience=30,
        )
        # Keep the better of the two for y
        yl, yb = _eval_1d(x_best, N, c_eval)
        if yl * yb < y_score:
            y_best = x_best

    # Build positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = x_best[i_flat]
    base_y = y_best[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
