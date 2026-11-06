import numpy as np
import time

_START = None
_HARD_LIMIT = 145.0  # seconds before the 150s kill; leave safety margin

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time()

def _fast_density_index(weights, values, feasible):
    """O(n) fallback: pick max value/weight among feasible."""
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(weights > 0, values / weights, 0.0)
    fd = np.where(feasible, density, -np.inf)
    return int(np.argmax(fd))

def _learned_score(weights, values):
    """
    Learn a linear score = a * value + b * weight from the items themselves.
    We fit a simple ridge-regression-like model that tries to predict a
    'true' score that balances value and weight, then use it to rank items.
    
    Strategy: fit value ~ w * weight + b * 1  (regress value on weight),
    then residual = value - (w*weight + b). Items with high residual are
    'better than expected for their weight' -- prefer them.
    Combined score = residual + alpha * value.
    """
    n = len(weights)
    if n < 4:
        return values.copy()
    
    w = weights
    v = values
    
    # OLS: v = slope * w + intercept
    w_mean = w.mean()
    v_mean = v.mean()
    denom = np.sum((w - w_mean)**2)
    if denom < 1e-12:
        slope = 0.0
        intercept = v_mean
    else:
        slope = np.sum((w - w_mean) * (v - v_mean)) / denom
        intercept = v_mean - slope * w_mean
    
    residual = v - (slope * w + intercept)
    # Normalize residual to be comparable in scale to value
    res_std = residual.std()
    if res_std > 1e-12:
        residual = residual / res_std
    
    # Combine: high residual (undervalued by weight) + scaled original value
    alpha = 0.5
    score = alpha * v + (1.0 - alpha) * residual
    return score

def _learned_greedy_pack(cap, weights, values, score):
    """
    Greedily pack using the learned score, but with a twist:
    among feasible items, pick the one with highest score.
    This is a single-pass O(n log n) simulation of the full remaining problem.
    Returns (total_value, first_item_index).
    """
    n = len(weights)
    cap = float(cap)
    taken = np.zeros(n, dtype=bool)
    total = 0.0
    first_idx = -1
    
    # Precompute score
    # Sort by score descending for efficiency
    order = np.argsort(-score, kind='stable')
    
    for idx in order:
        idx = int(idx)
        if taken[idx]:
            continue
        wi = float(weights[idx])
        if wi > cap:
            continue
        vi = float(values[idx])
        cap -= wi
        total += vi
        taken[idx] = True
        if first_idx < 0:
            first_idx = idx
        if cap <= 0:
            break
    
    return total, first_idx

def _density_greedy_pack(cap, weights, values):
    """Standard density greedy for comparison/fallback."""
    n = len(weights)
    cap = float(cap)
    taken = np.zeros(n, dtype=bool)
    total = 0.0
    first_idx = -1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(weights > 0, values / weights, 0.0)
    order = np.argsort(-density, kind='stable')
    
    for idx in order:
        idx = int(idx)
        if taken[idx]:
            continue
        wi = float(weights[idx])
        if wi > cap:
            continue
        vi = float(values[idx])
        cap -= wi
        total += vi
        taken[idx] = True
        if first_idx < 0:
            first_idx = idx
        if cap <= 0:
            break
    
    return total, first_idx

def _value_greedy_pack(cap, weights, values):
    """Pure value greedy for comparison."""
    n = len(weights)
    cap = float(cap)
    taken = np.zeros(n, dtype=bool)
    total = 0.0
    first_idx = -1
    
    order = np.argsort(-values, kind='stable')
    
    for idx in order:
        idx = int(idx)
        if taken[idx]:
            continue
        wi = float(weights[idx])
        if wi > cap:
            continue
        vi = float(values[idx])
        cap -= wi
        total += vi
        taken[idx] = True
        if first_idx < 0:
            first_idx = idx
        if cap <= 0:
            break
    
    return total, first_idx

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if remaining_capacity is None:
        return -1
    cap = float(remaining_capacity)
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    n = len(weights)
    if n == 0 or cap <= 0:
        return -1
    
    feasible = weights <= cap
    if not feasible.any():
        return -1
    
    elapsed = _now() - _START
    if elapsed > _HARD_LIMIT:
        return _fast_density_index(weights, values, feasible)
    
    # Try multiple global packing strategies, pick the first item from the
    # strategy that yields the highest total value for the full remaining problem.
    
    best_total = -np.inf
    best_first = -1
    
    # Strategy 1: Density greedy (baseline)
    t1, f1 = _density_greedy_pack(cap, weights, values)
    if t1 > best_total:
        best_total = t1
        best_first = f1
    
    # Strategy 2: Learned score greedy
    score = _learned_score(weights, values)
    t2, f2 = _learned_greedy_pack(cap, weights, values, score)
    if t2 > best_total:
        best_total = t2
        best_first = f2
    
    # Strategy 3: Value greedy
    t3, f3 = _value_greedy_pack(cap, weights, values)
    if t3 > best_total:
        best_total = t3
        best_first = f3
    
    # Strategy 4: Inverse-weight (lightest first) -- sometimes good for tight capacity
    n2 = len(weights)
    cap4 = float(cap)
    taken4 = np.zeros(n2, dtype=bool)
    total4 = 0.0
    first4 = -1
    order4 = np.argsort(weights, kind='stable')
    for idx in order4:
        idx = int(idx)
        if taken4[idx]:
            continue
        wi = float(weights[idx])
        if wi > cap4:
            continue
        vi = float(values[idx])
        cap4 -= wi
        total4 += vi
        taken4[idx] = True
        if first4 < 0:
            first4 = idx
        if cap4 <= 0:
            break
    if total4 > best_total:
        best_total = total4
        best_first = first4
    
    # Strategy 5: Mixed score = value - lambda * weight (lambda tuned)
    # Try a few lambda values
    for lam in [0.5, 1.0, 2.0, 5.0]:
        score5 = values - lam * weights
        t5, f5 = _learned_greedy_pack(cap, weights, values, score5)
        if t5 > best_total:
            best_total = t5
            best_first = f5
        if _now() - _START > _HARD_LIMIT:
            break
    
    if best_first < 0:
        best_first = _fast_density_index(weights, values, feasible)
    
    return int(best_first)
