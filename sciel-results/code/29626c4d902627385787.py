"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

CAPACITY = 25.0
N_ITEMS = 100


def _solve_lp_relaxation(prizes, weights, capacity, n_cons, n_items):
    """
    Solve the LP relaxation of the multiple-constraint knapsack.
    Maximize sum(c * x) subject to sum(A * x) <= b, 0 <= x <= 1.
    Returns dual prices (shadow prices) for each constraint.
    """
    # Objective: maximize sum(prizes * x) => minimize -sum(prizes * x)
    c_obj = -prizes.sum(axis=0)  # shape (n_items,)
    
    # Inequality constraints: weights @ x <= capacity for each constraint
    # A_ub has shape (n_cons, n_items), b_ub has shape (n_cons,)
    A_ub = weights  # shape (n_cons, n_items)
    b_ub = np.full(n_cons, capacity)
    
    # Bounds: 0 <= x_i <= 1
    bounds = [(0, 1)] * n_items
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and res.ineqlin is not None:
            # Dual prices for inequality constraints
            dual_prices = np.abs(res.ineqlin.marginals) if hasattr(res.ineqlin, 'marginals') else np.zeros(n_cons)
            # scipy's linprog returns marginals in res.ineqlin.marginals or res.ineq.marginals depending on version
            if hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
                dual_prices = np.abs(res.ineqlin.marginals)
            elif hasattr(res, 'ineq') and hasattr(res.ineq, 'marginals'):
                dual_prices = np.abs(res.ineq.marginals)
            else:
                dual_prices = np.zeros(n_cons)
            return dual_prices
        else:
            return np.zeros(n_cons)
    except Exception:
        return np.zeros(n_cons)


def _compute_reduced_costs(prizes, weights, dual_prices, n_cons, n_items):
    """
    Compute reduced cost for each item: prize_i - sum_c(dual_price_c * weight_ic)
    Items with high reduced cost are "profitable" relative to the LP optimum.
    """
    total_prize = prizes.sum(axis=0)  # shape (n_items,)
    # Weighted sum of weights by dual prices: shape (n_items,)
    weighted_weights = weights.T @ dual_prices  # (n_items,)
    reduced_cost = total_prize - weighted_weights
    return reduced_cost


def _simulate_ant(heuristic, prizes, weights, capacity, n_items, n_cons):
    """Simulate a single ant: add items that still fit in every constraint,
    choosing the highest heuristic among the feasible ones (greedy).
    Returns the total prize of the produced solution."""
    remaining = capacity - weights[:, 0]
    for c in range(1, n_cons):
        remaining = np.minimum(remaining, capacity - weights[:, c])
    remaining = np.maximum(remaining, 0.0)
    feasible = remaining >= 0
    if not np.any(feasible):
        return 0.0
    best = int(np.argmax(np.where(feasible, heuristic, -np.inf)))
    total = 0.0
    while True:
        if remaining[best] < 0:
            return total
        total += prizes[:, best].sum()
        remaining = np.minimum(remaining, capacity - weights[:, best])
        feasible = remaining >= 0
        if not np.any(feasible):
            return total
        best = int(np.argmax(np.where(feasible, heuristic, -np.inf)))


def _evaluate_heuristic(heuristic, prizes, weights, capacity, n_items, n_cons):
    """Expected total prize of a heuristic = mean prize over several simulated ants."""
    s = 0.0
    for _ in range(8):
        s += _simulate_ant(heuristic, prizes, weights, capacity, n_items, n_cons)
    return s / 8.0


def _compute_heuristics(prizes, weights, deadline):
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n_cons, n_items = prizes.shape
    eps = 1e-9

    # Step 1: Solve LP relaxation to get dual prices (shadow prices)
    dual_prices = _solve_lp_relaxation(prizes, weights, CAPACITY, n_cons, n_items)
    
    # Step 2: Compute reduced costs
    reduced_cost = _compute_reduced_costs(prizes, weights, dual_prices, n_cons, n_items)
    
    # Step 3: Build candidate heuristics based on reduced cost and other features
    total_prize = prizes.sum(axis=0)
    max_w = weights.max(axis=0)
    sum_w = weights.sum(axis=0)
    
    # Primary: reduced cost (positive means item is profitable relative to LP)
    rc_positive = np.maximum(reduced_cost, 0.0)
    rc_norm = rc_positive / (rc_positive.max() + eps) if rc_positive.max() > 0 else np.ones(n_items)
    
    # Secondary: standard ratio features as fallback/complement
    ratio_max = total_prize / (max_w + eps)
    ratio_sum = total_prize / (sum_w + eps)
    
    # Per-constraint bottleneck ratio
    per_cons_ratio = np.ones(n_items)
    for c in range(n_cons):
        r = prizes[c] / (weights[c] + eps)
        per_cons_ratio = np.minimum(per_cons_ratio, r)
    
    pool = [
        # LP-based heuristics
        rc_norm + 1e-9,
        rc_norm * 2 + 1e-9,
        rc_norm ** 2 + 1e-9,
        rc_norm * (1 + ratio_max / (ratio_max.max() + eps)) + 1e-9,
        rc_norm * per_cons_ratio + 1e-9,
        
        # Hybrid: LP reduced cost blended with ratio
        0.5 * rc_norm + 0.5 * (ratio_max / (ratio_max.max() + eps)) + 1e-9,
        0.7 * rc_norm + 0.3 * (ratio_max / (ratio_max.max() + eps)) + 1e-9,
        0.3 * rc_norm + 0.7 * (ratio_max / (ratio_max.max() + eps)) + 1e-9,
        
        # Fallback: pure ratio-based (in case LP fails)
        ratio_max,
        ratio_sum,
        per_cons_ratio,
    ]
    
    # Score every candidate by the prize it actually yields in the ACO process.
    best_h = pool[0].copy()
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

    # Local refinement: try scaling
    for mult in (0.5, 0.75, 1.25, 1.5, 2.0, 3.0, 4.0):
        if time.time() > deadline:
            break
        cand = best_h * mult
        cand = np.maximum(cand, 1e-9)
        sc = _evaluate_heuristic(cand, prizes, weights, CAPACITY, n_items, n_cons)
        if sc > best_score:
            best_score = sc
            best_h = cand

    # Try mixing best_h with other pool candidates
    for h in pool:
        if time.time() > deadline:
            break
        hh = np.asarray(h, dtype=np.float64)
        hh = np.where(np.isfinite(hh), hh, 1e-9)
        hh = np.maximum(hh, 1e-9)
        for alpha in (0.2, 0.4, 0.6, 0.8):
            if time.time() > deadline:
                break
            cand = alpha * best_h + (1 - alpha) * hh
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

        deadline = start_time + 290.0  # safety margin before the 330s kill
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
