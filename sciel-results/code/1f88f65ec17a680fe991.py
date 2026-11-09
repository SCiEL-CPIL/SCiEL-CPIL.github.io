import numpy as np
import time

def _run_aco_fast(d, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Fast ACO simulation. Returns the minimum number of bins used by any ant
    across all iterations.
    """
    n = len(d)
    tau = np.ones((n, n))
    best_overall = np.inf
    rng = np.random.default_rng(seed)
    
    # Pre-compute the valid mask for each possible current fill? No, fill changes.
    # But we can pre-compute d.
    
    for it in range(n_iters):
        ant_bins = []
        
        for ant in range(n_ants):
            used = np.zeros(n, dtype=bool)
            bins = 1
            current_item = -1
            current_fill = 0.0
            
            # Open first bin: pick a random fitting item
            # Since max demand < capacity, any item fits.
            # Optimization: Instead of random, maybe pick the largest? 
            # The task says "places a random fitting item". We must follow this.
            j = rng.integers(0, n)
            used[j] = True
            current_fill = d[j]
            current_item = j
            
            # Main loop
            while True:
                remaining = capacity - current_fill
                # Candidates: not used and d[j] <= remaining
                # Optimization: Pre-compute sorted indices? 
                # For n=120, simple boolean mask is fast enough.
                mask = (~used) & (d <= remaining)
                
                if not np.any(mask):
                    # Open new bin
                    bins += 1
                    unused_indices = np.where(~used)[0]
                    if len(unused_indices) == 0:
                        break
                    j = unused_indices[rng.integers(0, len(unused_indices))]
                    used[j] = True
                    current_fill = d[j]
                    current_item = j
                    continue
                else:
                    # Get probabilities
                    row_tau = tau[current_item, :]
                    row_heur = heuristic[current_item, :]
                    probs = row_tau * row_heur
                    probs[~mask] = 0.0
                    
                    total_prob = np.sum(probs)
                    if total_prob <= 0:
                        candidates = np.where(mask)[0]
                        j = candidates[rng.integers(0, len(candidates))]
                    else:
                        probs = probs / total_prob
                        cum_probs = np.cumsum(probs)
                        r = rng.random()
                        idx = np.searchsorted(cum_probs, r)
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
            
        # Pheromone update
        tau = tau * 0.9 + (1.0 / min_bins)
        
    return best_overall

def heuristics(demands, capacity):
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    start_time = time.time()
    time_limit = 320.0  # Safety margin
    
    # Pre-compute slack matrix
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    slack = np.where(valid, capacity - pair_sum, 1e9)
    
    best_score = np.inf
    best_heuristic = None
    
    # We will search over a parameterized family of heuristics.
    # The previous program used 1/slack^alpha.
    # Let's refine the search space.
    # Also, consider that "best fit" in bin packing often means minimizing the remaining space.
    # So 1/slack is good.
    # What about a combination of slack and size?
    # e.g. prefer items that are large (to reduce the number of items left)?
    # Or prefer items that are small (to fit more)?
    # Usually, larger items are harder to place, so placing them early is good.
    # But here the heuristic is for "placing j after i".
    # If i is large, j should be small to fit?
    # If i is small, j can be anything?
    
    # Let's try a grid of alpha values for 1/slack^alpha.
    # And also try a linear combination with a size-based term.
    # h[i,j] = (1/slack[i,j])^alpha * (d[j])^beta
    
    alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    betas = [0.0, 0.5, 1.0]
    
    # To save time, we will evaluate a subset of these with fewer seeds first,
    # then refine the best ones with more seeds.
    
    # Phase 1: Coarse search with 1 seed
    for alpha in alphas:
        if time.time() - start_time > time_limit:
            break
        for beta in betas:
            if time.time() - start_time > time_limit:
                break
            
            if alpha == 0:
                base = np.where(valid, 1.0, 1e-9)
            else:
                base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
            
            if beta > 0:
                base = base * (d[None, :]**beta)
            
            base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
            
            score = _run_aco_fast(d, capacity, base, n_ants=20, n_iters=30, seed=42)
            
            if score < best_score:
                best_score = score
                best_heuristic = base.copy()
    
    # Phase 2: Fine search around the best alpha/beta found in Phase 1
    # We need to track the best alpha/beta. Let's store them.
    # Let's re-run Phase 1 but store the best parameters.
    best_alpha = 1.0
    best_beta = 0.0
    best_score_coarse = np.inf
    
    for alpha in alphas:
        if time.time() - start_time > time_limit:
            break
        for beta in betas:
            if time.time() - start_time > time_limit:
                break
            
            if alpha == 0:
                base = np.where(valid, 1.0, 1e-9)
            else:
                base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
            
            if beta > 0:
                base = base * (d[None, :]**beta)
            
            base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
            
            score = _run_aco_fast(d, capacity, base, n_ants=20, n_iters=30, seed=42)
            
            if score < best_score_coarse:
                best_score_coarse = score
                best_alpha = alpha
                best_beta = beta
    
    # Now refine around best_alpha and best_beta with multiple seeds
    alpha_range = np.linspace(max(0.1, best_alpha - 0.5), best_alpha + 0.5, 5)
    beta_range = np.linspace(max(0.0, best_beta - 0.2), best_beta + 0.2, 3)
    
    for alpha in alpha_range:
        if time.time() - start_time > time_limit:
            break
        for beta in beta_range:
            if time.time() - start_time > time_limit:
                break
            
            if alpha == 0:
                base = np.where(valid, 1.0, 1e-9)
            else:
                base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
            
            if beta > 0:
                base = base * (d[None, :]**beta)
            
            base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
            
            # Evaluate with 3 seeds
            scores = []
            for seed in [42, 123, 456]:
                if time.time() - start_time > time_limit:
                    break
                score = _run_aco_fast(d, capacity, base, n_ants=20, n_iters=30, seed=seed)
                scores.append(score)
            
            if scores:
                avg_score = np.mean(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_heuristic = base.copy()
    
    if best_heuristic is None:
        # Fallback
        base = np.where(valid, 1.0 / (slack + 1e-9), 1e-9)
        best_heuristic = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
    
    return best_heuristic
