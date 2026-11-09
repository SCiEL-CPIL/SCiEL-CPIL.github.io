import numpy as np
import time

def _run_aco(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """Run ACO and return min bins used by any ant."""
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    tau = np.ones((n, n))
    best_overall = np.inf
    rng = np.random.default_rng(seed)
    
    for it in range(n_iters):
        ant_bins = []
        for ant in range(n_ants):
            used = np.zeros(n, dtype=bool)
            bins = 1
            current_item = -1
            current_fill = 0.0
            
            # First item: random
            j = rng.integers(0, n)
            used[j] = True
            current_fill = d[j]
            current_item = j
            
            while True:
                remaining = capacity - current_fill
                mask = (~used) & (d <= remaining)
                
                if not np.any(mask):
                    bins += 1
                    unused = np.where(~used)[0]
                    if len(unused) == 0:
                        break
                    j = unused[rng.integers(0, len(unused))]
                    used[j] = True
                    current_fill = d[j]
                    current_item = j
                    continue
                
                row_tau = tau[current_item, :]
                row_heur = heuristic[current_item, :]
                probs = row_tau * row_heur
                probs[~mask] = 0.0
                total = probs.sum()
                
                if total <= 0:
                    candidates = np.where(mask)[0]
                    j = candidates[rng.integers(0, len(candidates))]
                else:
                    probs = probs / total
                    cum = np.cumsum(probs)
                    r = rng.random()
                    idx = np.searchsorted(cum, r)
                    if idx >= n:
                        idx = n - 1
                    j = idx
                
                used[j] = True
                current_fill += d[j]
                current_item = j
            
            ant_bins.append(bins)
        
        min_bins = min(ant_bins)
        if min_bins < best_overall:
            best_overall = min_bins
        
        tau = tau * 0.9 + (1.0 / min_bins)
    
    return best_overall

def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    start_time = time.time()
    time_limit = 300.0
    
    # Precompute pair sums and validity
    pair = d[:, None] + d[None, :]
    valid = pair <= capacity
    slack = np.where(valid, capacity - pair, np.inf)
    ratio = np.where(valid, pair / capacity, 0.0)
    
    # Candidate heuristics to try
    candidates = []
    
    # 1. Tight fit: high when ratio close to 1
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        h = np.where(valid, ratio**alpha, 1e-9)
        candidates.append(h)
    
    # 2. Inverse slack: high when slack small
    for alpha in [0.5, 1.0, 2.0, 3.0]:
        h = np.where(valid, 1.0 / (slack + 1e-6)**alpha, 1e-9)
        candidates.append(h)
    
    # 3. Combined: ratio * inverse_slack
    for alpha in [0.5, 1.0, 2.0]:
        h = np.where(valid, ratio**alpha / (slack + 1e-6), 1e-9)
        candidates.append(h)
    
    # 4. Weighted by item sizes: prefer pairing large items tightly
    for alpha in [1.0, 2.0, 3.0]:
        size_weight = (d[:, None] + d[None, :])**alpha
        h = np.where(valid, size_weight * ratio**alpha, 1e-9)
        candidates.append(h)
    
    # 5. Pure best fit: 1/slack
    h = np.where(valid, 1.0 / (slack + 1e-6), 1e-9)
    candidates.append(h)
    
    # 6. Worst fit: slack (leave room)
    h = np.where(valid, slack, 1e-9)
    candidates.append(h)
    
    # Evaluate candidates with multiple seeds
    best_score = np.inf
    best_heuristic = None
    seeds = [42, 123, 456, 789, 1024]
    
    # Evaluate a subset first for speed
    num_to_try = min(len(candidates), 15)
    for idx in range(num_to_try):
        if time.time() - start_time > time_limit - 60:
            break
        
        h = candidates[idx]
        # Ensure valid: positive, finite, correct shape
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        assert h.shape == (n, n)
        
        scores = []
        for seed in seeds[:3]:  # Use 3 seeds for speed
            score = _run_aco(d, capacity, h, seed=seed)
            scores.append(score)
        
        avg_score = np.mean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_heuristic = h.copy()
    
    # If time allows, try remaining candidates
    for idx in range(num_to_try, len(candidates)):
        if time.time() - start_time > time_limit - 30:
            break
        
        h = candidates[idx]
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        
        scores = []
        for seed in seeds[:3]:
            score = _run_aco(d, capacity, h, seed=seed)
            scores.append(score)
        
        avg_score = np.mean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_heuristic = h.copy()
    
    # If no candidate worked, use default tight fit
    if best_heuristic is None:
        best_heuristic = np.where(valid, ratio**2, 1e-9)
        best_heuristic = np.where(np.isfinite(best_heuristic) & (best_heuristic > 0), best_heuristic, 1e-9)
    
    return best_heuristic
