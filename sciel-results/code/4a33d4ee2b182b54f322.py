import numpy as np
import time

def _ffd_bin_assignment(d, capacity):
    """Run FFD and return a list of bins (each bin is a list of indices)."""
    n = len(d)
    order = np.argsort(-d)  # descending
    bins = []  # list of (fill, [indices])
    for idx in order:
        placed = False
        for bi, (fill, members) in enumerate(bins):
            if d[idx] <= capacity - fill:
                bins[bi] = (fill + d[idx], members + [idx])
                placed = True
                break
        if not placed:
            bins.append((d[idx], [idx]))
    return [members for _, members in bins]

def _build_ffd_prior(d, capacity, n):
    """Build a sparse compatibility matrix from FFD bin assignments.
    Returns an n x n matrix where H[i,j] = 1 if i and j are in the same FFD bin, else 0.
    This is used as a structural prior to boost the heuristic for known-good pairings.
    """
    bins = _ffd_bin_assignment(d, capacity)
    prior = np.zeros((n, n))
    for members in bins:
        for i in members:
            for j in members:
                if i != j:
                    prior[i, j] = 1.0
    return prior

def _fast_aco_eval(demands, capacity, heuristic, n_ants=5, n_iters=10, seed=42):
    """
    Fast evaluation of the heuristic by running a truncated ACO simulation.
    Returns the best number of bins found.
    """
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    # Pheromone initialization
    tau = np.ones((n, n))
    
    best_overall = np.inf
    rng = np.random.default_rng(seed)
    
    for it in range(n_iters):
        min_bins_in_iter = np.inf
        
        for ant in range(n_ants):
            used = np.zeros(n, dtype=bool)
            bins = 1
            current_item = -1
            current_fill = 0.0
            
            # Place a random fitting item
            j = rng.integers(0, n)
            used[j] = True
            current_fill = d[j]
            current_item = j
            
            while True:
                remaining = capacity - current_fill
                # Candidates: not used and d[j] <= remaining
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
            
        min_bins_in_iter = min(min_bins_in_iter, bins)
        
        # Update pheromone
        # "every entry gains 1/(fewest bins of that iteration)"
        tau = tau * 0.9 + (1.0 / min_bins_in_iter)
        
        if min_bins_in_iter < best_overall:
            best_overall = min_bins_in_iter
            
    return best_overall

def heuristics(demands, capacity):
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    start_time = time.time()
    time_limit = 270.0  # Safety margin before 330s kill
    
    # Pre-compute pair sums and slack
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    slack = np.where(valid, capacity - pair_sum, 1e9)
    
    # Build FFD structural prior
    ffd_prior = _build_ffd_prior(d, capacity, n)
    
    best_score = np.inf
    best_alpha = 1.0
    best_beta = 0.5  # weight for FFD prior
    
    # Search over alpha (slack exponent) and beta (FFD prior weight)
    # H_ij = (1/slack^alpha) * (1 + beta * ffd_prior[i,j])
    # Coarse grid for alpha: 0.3 to 8.0
    alphas = np.linspace(0.3, 8.0, 12)
    betas = [0.0, 0.5, 1.0, 2.0]
    
    # To be robust against seed noise, use 3 seeds
    seeds = [42, 123, 7]
    
    for alpha in alphas:
        if time.time() - start_time > time_limit:
            break
        
        base_slack = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
        base_slack = np.where(np.isfinite(base_slack) & (base_slack > 0), base_slack, 1e-9)
        
        for beta in betas:
            if time.time() - start_time > time_limit:
                break
            
            # Combine slack-based heuristic with FFD prior
            # H = base_slack * (1 + beta * ffd_prior)
            base = base_slack * (1.0 + beta * ffd_prior)
            base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
            
            scores = []
            for seed in seeds:
                score = _fast_aco_eval(d, capacity, base, n_ants=5, n_iters=10, seed=seed)
                scores.append(score)
            
            # Use soft-min (log-sum-exp trick) for a smoother surrogate
            scores_arr = np.array(scores, dtype=float)
            max_s = np.max(scores_arr)
            # soft-min = -log(sum(exp(-s))) + max_s ... actually just use mean for now
            # but let's use a weighted mean that penalizes the worst seed more
            avg_score = np.mean(scores_arr)
            
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha
                best_beta = beta
    
    # Fine polish around best parameters
    if time.time() - start_time < time_limit - 15.0:
        center_alpha = best_alpha
        center_beta = best_beta
        span_a = 0.4
        span_b = 0.3
        
        fine_alphas = np.linspace(max(0.1, center_alpha - span_a), center_alpha + span_a, 5)
        fine_betas = np.linspace(max(0.0, center_beta - span_b), center_beta + span_b, 3)
        
        for alpha in fine_alphas:
            if time.time() - start_time > time_limit:
                break
            
            base_slack = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
            base_slack = np.where(np.isfinite(base_slack) & (base_slack > 0), base_slack, 1e-9)
            
            for beta in fine_betas:
                if time.time() - start_time > time_limit:
                    break
                
                base = base_slack * (1.0 + beta * ffd_prior)
                base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
                
                scores = []
                for seed in seeds:
                    score = _fast_aco_eval(d, capacity, base, n_ants=5, n_iters=10, seed=seed)
                    scores.append(score)
                
                avg_score = np.mean(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_alpha = alpha
                    best_beta = beta
    
    # High-precision polish: more ants and iterations with the best params
    if time.time() - start_time < time_limit - 30.0:
        base_slack = np.where(valid, 1.0 / (slack + 1e-9)**best_alpha, 1e-9)
        base_slack = np.where(np.isfinite(base_slack) & (base_slack > 0), base_slack, 1e-9)
        base = base_slack * (1.0 + best_beta * ffd_prior)
        base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
        
        # Try a few nearby alphas with more ants to break ties
        polish_alphas = [best_alpha, best_alpha * 0.9, best_alpha * 1.1]
        best_polish_score = best_score
        best_polish_alpha = best_alpha
        
        for pa in polish_alphas:
            if time.time() - start_time > time_limit:
                break
            if pa <= 0:
                continue
            
            ps_slack = np.where(valid, 1.0 / (slack + 1e-9)**pa, 1e-9)
            ps_slack = np.where(np.isfinite(ps_slack) & (ps_slack > 0), ps_slack, 1e-9)
            ps_base = ps_slack * (1.0 + best_beta * ffd_prior)
            ps_base = np.where(np.isfinite(ps_base) & (ps_base > 0), ps_base, 1e-9)
            
            scores = []
            for seed in seeds:
                score = _fast_aco_eval(d, capacity, ps_base, n_ants=8, n_iters=12, seed=seed)
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score < best_polish_score:
                best_polish_score = avg_score
                best_polish_alpha = pa
        
        best_alpha = best_polish_alpha
    
    # Construct final heuristic with best parameters
    final_slack = np.where(valid, 1.0 / (slack + 1e-9)**best_alpha, 1e-9)
    final_slack = np.where(np.isfinite(final_slack) & (final_slack > 0), final_slack, 1e-9)
    final_heuristic = final_slack * (1.0 + best_beta * ffd_prior)
    final_heuristic = np.where(np.isfinite(final_heuristic) & (final_heuristic > 0), final_heuristic, 1e-9)
    
    return final_heuristic
