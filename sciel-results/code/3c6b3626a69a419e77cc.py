"""Cost-aware acquisition for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# --- module-level state -------------------------------------------------
_t0 = time.monotonic()
_HARD_DEADLINE = _t0 + 300.0     # hard wall-clock cap for the whole run
_PER_CALL_CAP = 15.0             # never let one call eat the budget
_SCALE = 1.0 / 10.24             # maps [-5.12,5.12] -> [0,1]
_EPS = 1e-9

# cheap deterministic local-search grid (coarse, uniform over the box)
_GRIDS = []
for _n in (11, 15, 21, 29):
    _ax = np.linspace(-5.12, 5.12, _n)
    _gx, _gy = np.meshgrid(_ax, _ax, indexing="ij")
    _GRIDS.append(np.column_stack((_gx.ravel(), _gy.ravel())))


def _kernel_matrix(S):
    d2 = np.sum((S[:, None, :] - S[None, :, :]) ** 2, axis=2)
    v = np.sqrt(d2 + 1e-12) / 0.2
    return np.exp(-v) * (1.0 + v + (v * v) / 3.0)


def _cross_kernel(SG, S):
    d2 = np.sum((SG[:, None, :] - S[None, :, :]) ** 2, axis=2)
    v = np.sqrt(d2 + 1e-12) / 0.2
    return np.exp(-v) * (1.0 + v + (v * v) / 3.0)


def _pareto_scores(ei, cost, sigma, n):
    """Rank-based Pareto front selection.

    Two objectives: maximize EI, minimize cost.
    Non-dominated points get high scores [0.6, 1.0] based on rank within the
    non-dominated set (ranked by a combined EI/cost heuristic).
    Dominated points get low scores [0.0, 0.3].
    A small uncertainty bonus is added to encourage exploration.
    """
    ei = np.asarray(ei, dtype=float)
    cost = np.asarray(cost, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(ei)
    if n == 0:
        return np.zeros(0)

    # Normalize cost to [0,1] for fair comparison
    c_min, c_max = np.min(cost), np.max(cost)
    c_range = c_max - c_min + _EPS
    cost_norm = (cost - c_min) / c_range  # 0 = cheapest, 1 = most expensive

    # Dominance: point i dominates point j if ei[i] >= ei[j] AND cost[i] <= cost[j]
    # (strict in at least one dimension). We want to find the non-dominated set.
    # Vectorized approach: for each point, count how many others dominate it.
    # A point is non-dominated if no other point strictly dominates it.
    #
    # i dominates j: ei[i] > ei[j] AND cost[i] <= cost[j]  OR  ei[i] >= ei[j] AND cost[i] < cost[j]
    # Simpler: i dominates j if (ei[i] >= ei[j] AND cost[i] <= cost[j]) AND (ei[i] > ei[j] OR cost[i] < cost[j])

    # For efficiency with n=2000, use a vectorized but memory-conscious approach.
    # Compute dominance counts via broadcasting in chunks to avoid O(n^2) memory blowup.
    # Actually 2000x2000 = 4M booleans = ~4MB, which is fine.

    # ei[i] >= ei[j] for all i,j
    ei_ge = ei[:, None] >= ei[None, :]       # (n, n)
    cost_le = cost[:, None] <= cost[None, :]  # (n, n)
    ei_gt = ei[:, None] > ei[None, :]
    cost_lt = cost[:, None] < cost[None, :]

    # i dominates j if (ei[i] >= ei[j] AND cost[i] <= cost[j]) AND (ei[i] > ei[j] OR cost[i] < cost[j])
    dominates = (ei_ge & cost_le) & (ei_gt | cost_lt)  # (n, n): row i, col j -> i dominates j

    # A point j is dominated if there exists some i that dominates j
    # dominated[j] = any(dominates[:, j]) = any over axis 0
    dominated = np.any(dominates, axis=0)  # shape (n,)
    non_dominated = ~dominated
    n_nd = int(np.sum(non_dominated))

    scores = np.zeros(n, dtype=float)

    if n_nd == 0:
        # Fallback: all dominated (shouldn't happen), use EI/cost ratio
        ratio = ei / np.maximum(cost, _EPS)
        r_min, r_max = np.min(ratio), np.max(ratio)
        r_range = r_max - r_min + _EPS
        scores = 0.3 * (ratio - r_min) / r_range
    elif n_nd == n:
        # All non-dominated: rank by combined heuristic within the set
        # Use EI/cost ratio as the ranking key within the non-dominated set
        ratio = ei / np.maximum(cost, _EPS)
        r_min, r_max = np.min(ratio), np.max(ratio)
        r_range = r_max - r_min + _EPS
        # Map to [0.6, 1.0]
        scores = 0.6 + 0.4 * (ratio - r_min) / r_range
    else:
        # Mixed: non-dominated get [0.6, 1.0], dominated get [0.0, 0.3]
        # Within non-dominated: rank by EI/cost ratio
        nd_idx = np.where(non_dominated)[0]
        nd_ratio = ei[nd_idx] / np.maximum(cost[nd_idx], _EPS)
        r_min, r_max = np.min(nd_ratio), np.max(nd_ratio)
        r_range = r_max - r_min + _EPS
        nd_scores = 0.6 + 0.4 * (nd_ratio - r_min) / r_range
        scores[nd_idx] = nd_scores

        # Within dominated: rank by EI/cost ratio (lower priority)
        dom_idx = np.where(dominated)[0]
        if len(dom_idx) > 0:
            dom_ratio = ei[dom_idx] / np.maximum(cost[dom_idx], _EPS)
            dr_min, dr_max = np.min(dom_ratio), np.max(dom_ratio)
            dr_range = dr_max - dr_min + _EPS
            dom_scores = 0.3 * (dom_ratio - dr_min) / dr_range
            scores[dom_idx] = dom_scores

    # Add a small uncertainty bonus to encourage exploration in high-sigma regions
    s_max = np.max(sigma) + _EPS
    scores = scores + 0.05 * (sigma / s_max)

    return scores


def _local_refine_best(mu, sigma, f_best, cost, deadline):
    """Compute the best EI/cost score on deterministic grids via analytic GP.

    Returns the best local score (scalar) or 0.0 if no time remains.
    """
    if len(_GRIDS) == 0 or (deadline - time.monotonic()) < 0.3:
        return 0.0

    m = len(mu)
    S = mu * _SCALE
    K = _kernel_matrix(S)
    noise = 1e-6
    s_safe = np.where(sigma > _EPS, sigma, _EPS)
    std = np.sqrt(s_safe ** 2 + noise)
    y = (f_best - mu) / std
    ym = np.mean(y)
    ysd = np.std(y) + 1e-9
    y = (y - ym) / ysd

    try:
        L = np.linalg.cholesky(K + np.eye(m) * 1e-6)
        Kinv_y = np.linalg.solve(L.T, np.linalg.solve(L, y))
        Kinv = np.linalg.inv(K + np.eye(m) * 1e-6)
    except Exception:
        Kinv_y = np.linalg.solve(K + np.eye(m) * 1e-6, y)
        Kinv = np.linalg.inv(K + np.eye(m) * 1e-6)

    best_score = 0.0
    for grid in _GRIDS:
        if (deadline - time.monotonic()) < 0.5:
            break
        SG = grid * _SCALE
        kg = _cross_kernel(SG, S)                 # (G, m)
        mu_g = ym + ysd * (kg @ Kinv_y)
        tmp = kg @ Kinv
        var_g = np.maximum(1.0 - np.sum(tmp * kg, axis=1), 0.0)
        sigma_g = np.sqrt(var_g * (std ** 2) + noise)
        sg = np.maximum(sigma_g, _EPS)
        zg = (f_best - mu_g) / sg
        eig = (f_best - mu_g) * norm.cdf(zg) + sg * norm.pdf(zg)
        cg = 1.0 + np.mean(grid, axis=1) * _SCALE  # cost per grid point
        local = eig / np.maximum(cg, _EPS)
        lb = float(np.max(local))
        if lb > best_score:
            best_score = lb

    return best_score


def acquisition(mu, sigma, f_best, cost):
    # --- deadline guard: return a safe, finite, deterministic answer fast ---
    if time.monotonic() > _HARD_DEADLINE - 5.0:
        cost_safe = np.maximum(np.asarray(cost, dtype=float), _EPS)
        return np.ones(len(mu)) / cost_safe          # favour cheap points

    t_start = time.monotonic()
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    f_best = float(f_best)
    n = len(mu)

    if n == 0:
        return np.zeros(0)

    # 1) Compute EI for each candidate in the batch
    s = np.maximum(sigma, _EPS)
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)

    # 2) Rank-based Pareto front selection (structural change)
    scores = _pareto_scores(ei, cost, sigma, n)

    # 3) Deterministic local-search lift (only if budget allows)
    if (time.monotonic() - t_start) < _PER_CALL_CAP - 2.0:
        try:
            lb = _local_refine_best(mu, sigma, f_best, cost, _HARD_DEADLINE - 5.0)
            if lb > 0:
                # Add a small boost proportional to the best local score,
                # normalized by the current max score to keep things bounded.
                scores = scores + 0.05 * (lb / (np.max(scores) + _EPS))
        except Exception:
            pass

    # 4) finite / bounded output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.clip(scores, -1e6, 1e6)

    return scores
# EVOLVE-BLOCK-END
