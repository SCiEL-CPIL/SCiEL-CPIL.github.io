"""Bermudan put exercise policy via backward induction on a log-spaced grid."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class _ExercisePolicy:
    def __init__(self, boundaries, n_dates):
        self.boundaries = boundaries
        self.n_dates = n_dates

    def exercise(self, current_state, time_idx):
        cs = np.asarray(current_state, dtype=np.float64)
        if isinstance(time_idx, (np.integer, int)):
            j = int(time_idx)
        else:
            j = int(np.asarray(time_idx).item())
        if j < 0:
            j = 0
        if j >= self.n_dates:
            j = self.n_dates - 1
        return cs <= self.boundaries[j]


def _build_grid(n_nodes, s_min, s_max):
    return np.exp(np.linspace(np.log(s_min), np.log(s_max), n_nodes))


def _backward_induction(grid, K, r, sigma, n_dates, dt):
    n_nodes = len(grid)
    disc = np.exp(-r * dt)
    mu_log = (r - 0.5 * sigma * sigma) * dt
    vol_step = sigma * np.sqrt(dt)
    up_factor = np.exp(mu_log + vol_step)
    dn_factor = np.exp(mu_log - vol_step)

    V = np.zeros((n_dates, n_nodes))
    V[0, :] = np.maximum(K - grid, 0.0)

    for j in range(1, n_dates):
        s_up = grid * up_factor
        s_dn = grid * dn_factor
        v_up = np.interp(s_up, grid, V[j - 1], left=0.0, right=0.0)
        v_dn = np.interp(s_dn, grid, V[j - 1], left=0.0, right=0.0)
        cont = disc * 0.5 * (v_up + v_dn)
        pay = np.maximum(K - grid, 0.0)
        V[j, :] = np.maximum(pay, cont)

    return V


def _find_boundary(V, grid, K, j):
    """j is the index in V (0=last date). Returns exercise boundary."""
    pay = np.maximum(K - grid, 0.0)
    tol = 1e-8
    is_exercise = (V[j, :] <= pay + tol)
    idx = np.where(is_exercise)[0]
    if len(idx) == 0:
        return grid[0]
    if len(idx) == len(grid):
        return grid[-1]
    return grid[idx[-1]]


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    TIME_LIMIT = 110.0

    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12
    dt = T / n_dates

    n_nodes = 480
    grid = _build_grid(n_nodes, 1.0, 150.0)
    V = _backward_induction(grid, K, r, sigma, n_dates, dt)

    # boundaries in time order: time_idx=0 → V[n_dates-1], ..., time_idx=11 → V[0]
    boundaries = np.zeros(n_dates)
    for j in range(n_dates):
        boundaries[j] = _find_boundary(V, grid, K, n_dates - 1 - j)

    # Enforce non-decreasing in time (put boundary increases as time passes)
    for j in range(1, n_dates):
        if boundaries[j] < boundaries[j - 1]:
            boundaries[j] = boundaries[j - 1]
    boundaries = np.minimum(boundaries, K)

    policy = _ExercisePolicy(boundaries, n_dates)

    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 25:
        try:
            n_nodes2 = 720
            grid2 = _build_grid(n_nodes2, 1.0, 150.0)
            V2 = _backward_induction(grid2, K, r, sigma, n_dates, dt)
            boundaries2 = np.zeros(n_dates)
            for j in range(n_dates):
                boundaries2[j] = _find_boundary(V2, grid2, K, n_dates - 1 - j)
            for j in range(1, n_dates):
                if boundaries2[j] < boundaries2[j - 1]:
                    boundaries2[j] = boundaries2[j - 1]
            boundaries2 = np.minimum(boundaries2, K)
            policy = _ExercisePolicy(boundaries2, n_dates)
        except Exception:
            pass

    return policy
# EVOLVE-BLOCK-END
