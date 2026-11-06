import numpy as np
import time

# Global deadline guard: total wall-clock budget for the whole evaluation.
_START = time.monotonic()
_TOTAL_BUDGET = 115.0  # leave margin before the 120s/150s kill

def _remaining_budget():
    return max(0.0, _TOTAL_BUDGET - (time.monotonic() - _START))

def _greedy_complete(weights, values, capacity, policy='density'):
    """Simulate greedy completion from a state. Returns total added value."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    
    if policy == 'density':
        order = np.argsort(-v / np.maximum(w, 1e-12))
    elif policy == 'value':
        order = np.argsort(-v)
    elif policy == 'blend':
        order = np.argsort(-(v / np.maximum(w, 1e-12) + 0.15 * v))
    else:
        order = np.argsort(-v / np.maximum(w, 1e-12))
    
    cap = float(capacity)
    total = 0.0
    for i in order:
        wi = float(w[i])
        if wi <= cap:
            total += float(v[i])
            cap -= wi
            if cap <= 1e-9:
                break
    return total

def _score_candidates(w, v, cap, n):
    """Generate diverse candidate indices using multiple scoring families."""
    safe_w = np.maximum(w, 1e-12)
    density = v / safe_w
    
    # Multiple scoring families
    scores = np.vstack([
        v,                                      # pure value
        density,                                # pure density
        density + 0.1 * v,                      # blend
        v - 0.02 * w,                           # value minus weight penalty
        density * (1.0 + 0.3 * (w / cap)),      # density boosted by relative size
        v / (w + 0.1 * cap),                    # smoothed density
    ])
    
    cands = set()
    k = min(n, 10)
    for s in scores:
        top = np.argsort(-s)[:k]
        for t in top:
            cands.add(int(t))
    
    # Ensure density-argmax and value-argmax are included
    cands.add(int(np.argmax(density)))
    cands.add(int(np.argmax(v)))
    
    return list(cands), density

def _evaluate_candidate(ci, w, v, cap, n, density):
    """Estimate total value if we pick candidate ci now, using multi-policy greedy completion."""
    used_w = float(w[ci])
    used_v = float(v[ci])
    new_cap = cap - used_w
    if new_cap < 0:
        return -np.inf
    
    # Build mask excluding candidate
    mask = np.ones(n, dtype=bool)
    mask[ci] = False
    rw = w[mask]
    rv = v[mask]
    
    # Try multiple completion policies and take the max
    est = used_v
    for pol in ('density', 'value', 'blend'):
        est = max(est, used_v + _greedy_complete(rw, rv, new_cap, pol))
    return est

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    cap = float(remaining_capacity)
    if cap <= 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    n = len(w)
    if n == 0:
        return -1
    
    # Only consider feasible items
    feasible = w <= cap
    if not np.any(feasible):
        return -1
    
    # Fast path if near deadline
    if _remaining_budget() < 0.1:
        feasible_v = np.where(feasible, v / np.maximum(w, 1e-12), -np.inf)
        idx = int(np.argmax(feasible_v))
        return idx if feasible[idx] else -1
    
    # Generate diverse candidates
    cands, density = _score_candidates(w, v, cap, n)
    
    # Filter to feasible candidates only
    cands = [c for c in cands if feasible[c]]
    if not cands:
        return -1
    
    # Limit candidates to keep per-call cost bounded
    # Use a slightly larger limit to allow better exploration, but still bounded
    sim_limit = 50
    if len(cands) > sim_limit:
        cands.sort(key=lambda i: -density[i])
        cands = cands[:sim_limit]
    
    # Evaluate each candidate with bounded lookahead
    best_idx = cands[0]
    best_est = -np.inf
    
    for ci in cands:
        est = _evaluate_candidate(ci, w, v, cap, n, density)
        if est > best_est:
            best_est = est
            best_idx = ci
        
        # Deadline check
        if _remaining_budget() < 0.5:
            break
    
    return int(best_idx)
