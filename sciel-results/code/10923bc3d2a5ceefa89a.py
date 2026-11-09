# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 135.0
_DEADLINE_CHECK_EVERY = 50000
_CACHE = {}


def _sim_score(item, bins, params):
    """Vectorized score for one item across all bins (higher = more preferred).

    params: (p, a, b)
      p : exponent on remaining capacity (larger => prefer emptier bins / worst-fit)
      a : weight of the 'fit' component  (push toward filling the bin)
      b : weight of the 'balance' component (prefer emptier bins overall)
    """
    p, a, b = params
    rem = bins - item
    cap = bins.max() + 1e-9
    fr = np.clip(rem / cap, 0.0, 1.0)
    score = a * fr ** p + b * (1.0 - fr)
    return score


def _simulate_fast(items, cap, params):
    """Fast online bin-packing simulation. Returns (num_bins_used, num_items_placed)."""
    n = len(items)
    max_bins = n + 1
    bins = np.full((max_bins,), cap, dtype=np.float64)
    used = 0
    placed = 0
    p, a, b = params
    for i in range(n):
        if i % _DEADLINE_CHECK_EVERY == 0 and time.time() > _DEADLINE:
            return used, placed
        item = items[i]
        if item > cap:
            continue
        if used == 0:
            used = 1
            bins[0] = cap - item
            placed += 1
            continue
        rem = bins[:used] - item
        mask = rem >= -1e-9
        if not mask.any():
            used += 1
            bins[used - 1] = cap - item
            placed += 1
            continue
        # Compute scores only on feasible bins for speed
        feasible_bins = bins[:used][mask]
        feasible_rem = rem[mask]
        fr = feasible_rem / (feasible_bins.max() + 1e-9)
        fr = np.clip(fr, 0.0, 1.0)
        scores = a * fr ** p + b * (1.0 - fr)
        j = np.argmax(scores)
        # Map back to original index
        feasible_indices = np.where(mask)[0]
        bins[feasible_indices[j]] -= item
        placed += 1
    return used, placed


def _l1_lb(items, cap):
    """L1 lower bound: ceil(sum(items) / cap)."""
    return int(np.ceil(sum(items) / cap))


def _generate_weibull(n, cap, seed):
    """Generate a Weibull-distributed item sequence scaled so items fit in cap."""
    rng = np.random.RandomState(seed)
    shape, scale = 0.8, 15.0
    items = rng.weibull(shape, n) * scale
    items = np.clip(items, 0.05 * cap, cap * 0.95)
    return items


def _optimize_params():
    """Two-stage parameter search: coarse grid then local refinement."""
    n = 500
    cap = 100.0
    items = _generate_weibull(n, cap, 1234)
    lb = _l1_lb(items, cap)

    best_score = np.inf
    best_params = (1.0, 1.0, 0.2)

    # Stage 1: Coarse grid emphasizing worst-fit (high p, high b)
    p_grid = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    a_grid = [0.1, 0.5, 1.0, 1.5]
    b_grid = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]

    for p in p_grid:
        for a in a_grid:
            for b in b_grid:
                if time.time() > _DEADLINE - 30.0:
                    break
                used, placed = _simulate_fast(items, cap, (p, a, b))
                if placed == 0 or used == 0:
                    continue
                excess = (used - lb) / lb
                if excess < best_score:
                    best_score = excess
                    best_params = (p, a, b)
        if time.time() > _DEADLINE - 30.0:
            break

    # Stage 2: Local refinement around best point
    if time.time() < _DEADLINE - 20.0:
        p0, a0, b0 = best_params
        p_fine = [p0 * 0.5, p0 * 0.75, p0, p0 * 1.25, p0 * 1.5, p0 * 2.0]
        a_fine = [a0 * 0.5, a0 * 0.75, a0, a0 * 1.25, a0 * 1.5]
        b_fine = [b0 * 0.5, b0 * 0.75, b0, b0 * 1.25, b0 * 1.5, b0 * 2.0]
        for p in p_fine:
            if p <= 0:
                continue
            for a in a_fine:
                if a <= 0:
                    continue
                for b in b_fine:
                    if b < 0:
                        continue
                    if time.time() > _DEADLINE - 10.0:
                        break
                    used, placed = _simulate_fast(items, cap, (p, a, b))
                    if placed == 0 or used == 0:
                        continue
                    excess = (used - lb) / lb
                    if excess < best_score:
                        best_score = excess
                        best_params = (p, a, b)
                if time.time() > _DEADLINE - 10.0:
                    break
            if time.time() > _DEADLINE - 10.0:
                break

    return best_params


# ---- Startup: fit parameters once, cache the result ----
_PARAMS = _optimize_params()


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Return one score per bin with enough remaining capacity (higher = chosen)."""
    if not isinstance(bins, np.ndarray):
        bins = np.asarray(bins, dtype=np.float64)
    rem = bins - item
    scores = _sim_score(item, bins, _PARAMS)
    scores = np.where(rem >= -1e-9, scores, -np.inf)
    return scores
# EVOLVE-BLOCK-END
