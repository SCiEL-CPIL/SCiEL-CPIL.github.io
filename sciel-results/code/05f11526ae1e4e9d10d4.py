"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

CAPACITY = 25.0
N_ITEMS = 100


def _simulate_ant_fast(heuristic, prizes, weights, capacity, n_items, n_cons):
    """Fast greedy ant using heap for O(n log n) instead of O(n^2)."""
    # Precompute per-item total prize
    item_prizes = prizes.sum(axis=0)  # (n_items,)
    
    # For each item, compute how much "slack" it would leave in the tightest constraint
    # An item is feasible if for all constraints c: weights[c, i] <= remaining[c]
    # We track remaining capacity per constraint
    
    remaining = np.full(n_cons, capacity, dtype=np.float64)
    
    # Build heap: (-heuristic, item_index) for max-heap
    heap = [(-heuristic[i], i) for i in range(n_items)]
    heapq.heapify(heap)
    
    total = 0.0
    selected = np.zeros(n_items, dtype=bool)
    
    while heap:
        neg_h, idx = heapq.heappop(heap)
        if selected[idx]:
            continue
        # Check feasibility
        wi = weights[:, idx]
        if np.any(wi > remaining):
            continue
        # Select this item
        selected[idx] = True
        total += item_prizes[idx]
        remaining -= wi
        # Now we could re-push items that were previously infeasible but might be now
        # Actually in greedy knapsack, once we pick an item, remaining only decreases,
        # so no previously infeasible item becomes feasible. We just continue popping.
    
    return total


def _evaluate_heuristic(heuristic, prizes, weights, capacity, n_items, n_cons):
    """Expected total prize of a heuristic = mean prize over several simulated ants."""
    s = 0.0
    for _ in range(6):
        s += _simulate_ant_fast(heuristic, prizes, weights, capacity, n_items, n_cons)
    return s / 6.0


def _compute_heuristics(prizes, weights, deadline):
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n_cons, n_items = prizes.shape

    total_prize = prizes.sum(axis=0)          # (N,)
    sum_w = weights.sum(axis=0)               # (N,)
    max_w = weights.max(axis=0)               # (N,)
    min_w = weights.min(axis=0)               # (N,)
    # Effective weight: how much of the capacity does this item consume in the binding constraint
    # For multi-constraint, the binding constraint is the one where the item uses the most fraction
    # of capacity. Since all capacities are equal (25), binding = max weight across constraints.
    eps = 1e-9

    # Core heuristic families emphasizing max_w (binding constraint)
    ratio_max = total_prize / (max_w + eps)
    ratio_sum = total_prize / (sum_w + eps)
    ratio_min = total_prize / (min_w + eps)

    # Diversity: penalize items that are "large" in many constraints (harder to pack)
    # Number of constraints where the item is relatively large
    w_frac = weights / CAPACITY  # (n_cons, n_items), each in [0,1]
    n_large = np.sum(w_frac > 0.5, axis=0).astype(np.float64)  # (N,)
    pack_penalty = 1.0 / (1.0 + 0.5 * n_large)

    # Variance across constraints: items with balanced weights are more predictable
    w_var = np.var(weights, axis=0)
    balance_bonus = 1.0 / (1.0 + w_var)

    pool = [
        # Primary: ratio to max weight (binding constraint)
        ratio_max,
        ratio_max * pack_penalty,
        ratio_max * balance_bonus,
        ratio_max ** 1.3,
        ratio_max ** 1.5,
        ratio_max * (1.0 + 0.1 * total_prize),
        # Sum-based variants
        ratio_sum,
        ratio_sum * pack_penalty,
        ratio_sum ** 1.3,
        # Min-based (least constraining)
        ratio_min,
        ratio_min * pack_penalty,
        # Hybrid: geometric mean of max and sum ratios
        np.sqrt(ratio_max * ratio_sum),
        np.sqrt(ratio_max * ratio_sum) * pack_penalty,
        # Prize-weighted with exponential decay on weight
        total_prize * np.exp(-0.3 * max_w),
        total_prize * np.exp(-0.5 * max_w),
        total_prize * np.exp(-0.2 * sum_w),
        # Square prize over max weight
        total_prize ** 2 / (max_w + eps),
        # Inverse weight emphasis
        1.0 / (max_w + eps) * total_prize,
        # Combined: prize density with packing difficulty
        ratio_max / (1.0 + 0.3 * n_large),
        ratio_max * np.exp(-0.2 * n_large),
    ]

    # Score every candidate by the prize it actually yields in the ACO process.
    best_h = pool[0]
    best_score = -np.inf
    for h in pool:
        if time.time() > deadline:
            break
        hh = np.asarray(h, dtype=np.float64)
        hh = np.where(np.isfinite(hh), hh, 1e-9)
        hh = np.maximum(hh, 1e-9)
        sc = _evaluate_heuristic(hh, prizes, weights, CAPACITY, n_items, n_cons)
        if sc > best_score:
            best_score = sc
            best_h = hh.copy()

    # Local refinement: try scaling and small perturbations
    for mult in (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0):
        if time.time() > deadline:
            break
        cand = best_h * mult
        cand = np.maximum(cand, 1e-9)
        sc = _evaluate_heuristic(cand, prizes, weights, CAPACITY, n_items, n_cons)
        if sc > best_score:
            best_score = sc
            best_h = cand

    # Try blending top-2 candidates
    if time.time() < deadline - 10:
        # Re-score top candidates and try blends
        scored = []
        for h in pool[:8]:
            if time.time() > deadline:
                break
            hh = np.asarray(h, dtype=np.float64)
            hh = np.where(np.isfinite(hh), hh, 1e-9)
            hh = np.maximum(hh, 1e-9)
            sc = _evaluate_heuristic(hh, prizes, weights, CAPACITY, n_items, n_cons)
            scored.append((sc, hh))
        scored.sort(key=lambda x: -x[0])
        if len(scored) >= 2:
            s1, h1 = scored[0]
            s2, h2 = scored[1]
            for alpha in (0.3, 0.5, 0.7):
                if time.time() > deadline:
                    break
                cand = alpha * h1 + (1 - alpha) * h2
                cand = np.maximum(cand, 1e-9)
                sc = _evaluate_heuristic(cand, prizes, weights, CAPACITY, n_items, n_cons)
                if sc > best_score:
                    best_score = sc
                    best_h = cand

    best_h = np.where(np.isfinite(best_h), best_h, 1e-9)
    best_h = np.maximum(best_h, 1e-9)
    return best_h


def heuristics(prizes, weights):
    start_time = time.time()
    try:
        p = np.asarray(prizes, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        if p.ndim != 2 or w.ndim != 2:
            raise ValueError("bad shape")
        if p.shape != w.shape:
            raise ValueError("shape mismatch")
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(w)):
            raise ValueError("non-finite")
        n_cons, n_items = p.shape

        deadline = start_time + 295.0  # safety margin before the 330s kill
        result = _compute_heuristics(p, w, deadline)
        result = np.asarray(result, dtype=np.float64)
        if result.ndim != 1 or result.shape[0] != n_items:
            raise ValueError("bad result shape")
        if not np.all(np.isfinite(result)):
            raise ValueError("non-finite result")
        return result
    except Exception:
        try:
            p = np.asarray(prizes, dtype=np.float64)
            n_items = p.shape[1] if p.ndim == 2 else len(p)
        except Exception:
            n_items = N_ITEMS
        return np.ones(n_items, dtype=np.float64)
# EVOLVE-BLOCK-END
