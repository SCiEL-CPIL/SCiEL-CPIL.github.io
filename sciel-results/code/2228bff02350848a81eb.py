"""Bounded lookahead 0/1 knapsack heuristic with multi-policy greedy completion."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.monotonic()
_TOTAL_BUDGET = 115.0

def _remaining_budget():
    return max(0.0, _TOTAL_BUDGET - (time.monotonic() - _START))

def _greedy_complete(w, v, cap, policy='density'):
    """Simulate greedy completion. Returns total added value."""
    n = len(w)
    if n == 0 or cap <= 0:
        return 0.0
    if policy == 'density':
        order = np.argsort(-v / np.maximum(w, 1e-12))
    elif policy == 'value':
        order = np.argsort(-v)
    elif policy == 'blend':
        order = np.argsort(-(v / np.maximum(w, 1e-12) + 0.15 * v))
    else:
        order = np.argsort(-v / np.maximum(w, 1e-12))
    
    rem = float(cap)
    total = 0.0
    for i in order:
        wi = w[i]
        if wi <= rem:
            total += v[i]
            rem -= wi
            if rem <= 1e-9:
                break
    return total

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    cap = float(remaining_capacity)
    if cap <= 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    n = len(w)
    if n == 0:
        return -1
    
    feasible = w <= cap
    if not np.any(feasible):
        return -1
    
    # Fast path near deadline
    if _remaining_budget() < 0.1:
        safe_w = np.maximum(w, 1e-12)
        density = np.where(feasible, v / safe_w, -np.inf)
        idx = int(np.argmax(density))
        return idx if feasible[idx] else -1
    
    # Generate diverse candidate set
    safe_w = np.maximum(w, 1e-12)
    density = v / safe_w
    frac = w / cap if cap > 0 else np.zeros(n)
    
    scores = np.vstack([
        v,                                    # pure value
        density,                              # pure density
        density + 0.1 * v,                    # blend
        v / (w + 0.1 * cap),                  # smoothed density
        density * (1.0 + 0.3 * frac),         # density boosted by relative size
    ])
    
    cands = set()
    k = min(n, 8)
    for s in scores:
        top = np.argsort(-s)[:k]
        for t in top:
            if feasible[t]:
                cands.add(int(t))
    
    cands.add(int(np.argmax(np.where(feasible, density, -np.inf))))
    cands.add(int(np.argmax(np.where(feasible, v, -np.inf))))
    
    cands = [c for c in cands if feasible[c]]
    if not cands:
        return -1
    
    # Limit candidates for bounded cost
    if len(cands) > 30:
        cands.sort(key=lambda i: -density[i])
        cands = cands[:30]
    
    # Evaluate each candidate with multi-policy lookahead
    best_idx = cands[0]
    best_est = -np.inf
    
    for ci in cands:
        used_w = float(w[ci])
        used_v = float(v[ci])
        new_cap = cap - used_w
        if new_cap < 0:
            continue
        
        mask = np.ones(n, dtype=bool)
        mask[ci] = False
        rw = w[mask]
        rv = v[mask]
        
        # Multi-policy completion estimate
        est = used_v + _greedy_complete(rw, rv, new_cap, 'density')
        est = max(est, used_v + _greedy_complete(rw, rv, new_cap, 'value'))
        est = max(est, used_v + _greedy_complete(rw, rv, new_cap, 'blend'))
        
        if est > best_est:
            best_est = est
            best_idx = ci
        
        if _remaining_budget() < 0.5:
            break
    
    return int(best_idx)
# EVOLVE-BLOCK-END
