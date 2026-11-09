import numpy as np
import time

def _run_aco_fast(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Fast ACO simulation using precomputed candidate lists and vectorized operations.
    Returns the minimum number of bins used by any ant.
    Optimized for speed: uses local variables and minimal overhead.
    """
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    # Precompute: for each item i, which items j can follow i (d[i] + d[j] <= capacity)
    pair_sum = d[:, None] + d[None, :]
    valid_mask = pair_sum <= capacity
    
    # Precompute candidate indices for each item
    candidates_list = [np.where(valid_mask[i])[0] for i in range(n)]
    
    # Pheromone initialization
    tau = np.ones((n, n))
    
    best_overall = np.inf
    rng = np.random.default_rng(seed)
    
    for it in range(n_iters):
        min_bins_in_iter = n + 1 # Max possible is n
        
        for ant in range(n_ants):
            used = np.zeros(n, dtype=bool)
            bins = 1
            # Open first bin - place random item
            j = rng.integers(0, n)
            used[j] = True
            current_item = j
            
            # Place remaining items
            while True:
                cands = candidates_list[current_item]
                if len(cands) == 0:
                    break
                valid_cands = cands[~used[cands]]
                if len(valid_cands) == 0:
                    bins += 1
                    if bins > min_bins_in_iter:
                        # Early termination for this ant if already worse than best in this iter
                        # But we need to finish to update used? No, we just need the count.
                        # We can break early if we just want the count, but we need to place items to get the count?
                        # Actually, if we break, we don't know the final count.
                        # However, if bins is already high, it won't be the minimum.
                        # But we must place all items to get the true count.
                        # So we can't break the placement loop, but we can skip probability calc if we are already bad?
                        # No, we must follow the algorithm.
                        pass
                    
                    unused_indices = np.where(~used)[0]
                    if len(unused_indices) == 0:
                        break
                    j = unused_indices[rng.integers(0, len(unused_indices))]
                    used[j] = True
                    current_item = j
                    continue
                
                # Compute probabilities
                row_tau = tau[current_item, valid_cands]
                row_heur = heuristic[current_item, valid_cands]
                probs = row_tau * row_heur
                
                total_prob = np.sum(probs)
                if total_prob <= 0:
                    j = valid_cands[rng.integers(0, len(valid_cands))]
                else:
                    probs /= total_prob
                    r = rng.random()
                    cum_probs = np.cumsum(probs)
                    idx = np.searchsorted(cum_probs, r)
                    if idx >= len(valid_cands):
                        idx = len(valid_cands) - 1
                    j = valid_cands[idx]
                
                used[j] = True
                current_item = j

            if bins < min_bins_in_iter:
                min_bins_in_iter = bins
        
        if min_bins_in_iter < best_overall:
            best_overall = min_bins_in_iter
            
        tau = tau * 0.9 + (1.0 / min_bins_in_iter)
        
    return best_overall

def heuristics(demands, capacity):
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    start_time = time.time()
    # Conservative time limit to ensure we return within 300s
    time_limit = 280.0
    
    # Pre-compute matrices
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    slack = np.where(valid, capacity - pair_sum, 1e9)
    
    best_score = np.inf
    best_heuristic = None
    
    # Use faster, lower-fidelity ACO for tuning
    # n_ants=5, n_iters=5 is ~1/36th the cost of 20x30
    def evaluate_heuristic(base):
        score = _run_aco_fast(d, capacity, base, n_ants=5, n_iters=5, seed=42)
        return score
    
    def try_heuristic(base):
        nonlocal best_score, best_heuristic
        base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
        score = evaluate_heuristic(base)
        if score < best_score:
            best_score = score
            best_heuristic = base.copy()
        return score

    # Generate FFD bin structure to create a structural prior
    # Sort items in descending order
    sorted_indices = np.argsort(d)[::-1]
    ffd_bins = []
    current_bin = []
    current_sum = 0.0
    for idx in sorted_indices:
        item_size = d[idx]
        if current_sum + item_size <= capacity:
            current_bin.append(idx)
            current_sum += item_size
        else:
            if current_bin:
                ffd_bins.append(current_bin)
            current_bin = [idx]
            current_sum = item_size
    if current_bin:
        ffd_bins.append(current_bin)
    
    # Create a matrix marking pairs that co-occur in FFD bins
    ffd_cooccur = np.zeros((n, n), dtype=bool)
    for b in ffd_bins:
        for i in b:
            for j in b:
                if i != j:
                    ffd_cooccur[i, j] = True

    # Combine best-fit slack heuristic with FFD structural prior
    # The FFD prior boosts attractiveness for pairs known to work well together
    alphas = [1.5, 2.0, 2.5]
    ffd_weights = [0.0, 0.5, 1.0, 2.0]  # Weight of FFD boost
    
    for alpha in alphas:
        if time.time() - start_time > time_limit:
            break
        for ffd_w in ffd_weights:
            if time.time() - start_time > time_limit:
                break
            # Base best-fit heuristic
            base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
            # Add FFD structural boost: pairs in same FFD bin get extra attractiveness
            if ffd_w > 0:
                # Boost is multiplicative to preserve relative ordering
                boost = np.where(ffd_cooccur, 1.0 + ffd_w, 1.0)
                base = base * boost
            try_heuristic(base)
        
    # If we have time, test a couple of other simple heuristics
    if time.time() - start_time < time_limit - 50:
        # Worst fit with FFD prior
        for ffd_w in [1.0, 2.0]:
            if time.time() - start_time > time_limit:
                break
            base = np.where(valid, (slack + 1e-9)**1.0, 1e-9)
            if ffd_w > 0:
                boost = np.where(ffd_cooccur, 1.0 + ffd_w, 1.0)
                base = base * boost
            try_heuristic(base)
        
        # Target gap (0.8) with FFD prior
        for ffd_w in [1.0, 2.0]:
            if time.time() - start_time > time_limit:
                break
            target = capacity * 0.8
            dist = np.abs(pair_sum - target)
            base = np.where(valid, 1.0 / (dist + 1e-9), 1e-9)
            if ffd_w > 0:
                boost = np.where(ffd_cooccur, 1.0 + ffd_w, 1.0)
                base = base * boost
            try_heuristic(base)

    if best_heuristic is None:
        # Fallback to standard best-fit
        base = np.where(valid, 1.0 / (slack + 1e-9), 1e-9)
        best_heuristic = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
    
    return best_heuristic
