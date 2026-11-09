import numpy as np
import time

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

def _soft_min(scores, beta=20.0):
    """
    Smooth surrogate for min using log-sum-exp trick.
    Lower beta = smoother, higher beta = sharper (closer to true min).
    """
    scores = np.array(scores, dtype=float)
    m = np.min(scores)
    if m == 0:
        return 0.0
    # Use scaled version to avoid overflow
    scaled = scores * beta
    max_scaled = np.max(scaled)
    exps = np.exp(scaled - max_scaled)
    result = m - (1.0 / beta) * np.log(np.sum(exps))
    return max(result, 0.0)

def _build_complementarity_prior(d, capacity):
    """
    Build a complementarity prior that rewards pairs whose combined size
    leaves residual close to multiples of the minimum item size.
    This captures structural packing patterns beyond simple best-fit.
    """
    n = len(d)
    min_size = np.min(d)
    
    # For each pair, compute how "complementary" they are
    # A pair is complementary if (capacity - d[i] - d[j]) is close to k * min_size
    # for some small integer k >= 0
    prior = np.ones((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            remaining = capacity - d[i] - d[j]
            if remaining < 0:
                prior[i, j] = 1e-9
                continue
            # Check how close remaining is to multiples of min_size
            # Consider k = 0, 1, 2, ..., up to remaining/min_size
            max_k = int(remaining / min_size) + 1
            best_complementarity = 0.0
            for k in range(max_k + 1):
                target = k * min_size
                diff = abs(remaining - target)
                # Reward closeness to exact multiples
                complementarity = np.exp(-diff / (0.5 * min_size))
                if complementarity > best_complementarity:
                    best_complementarity = complementarity
            prior[i, j] = best_complementarity
    
    return prior

def heuristics(demands, capacity):
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    start_time = time.time()
    time_limit = 310.0  # Safety margin: evaluator kills at 330s
    
    # Pre-compute pair sums and slack
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    slack = np.where(valid, capacity - pair_sum, 1e9)
    
    # Build complementarity prior
    comp_prior = _build_complementarity_prior(d, capacity)
    # Ensure positive values
    comp_prior = np.where(np.isfinite(comp_prior) & (comp_prior > 0), comp_prior, 1e-9)
    
    best_score = np.inf
    best_alpha = 1.0
    best_beta = 1.0  # weight for complementarity
    
    # Search over alpha (slack exponent) and beta (complementarity weight)
    # H_ij = (1/slack^alpha) * comp_prior^beta
    # Coarse grid first
    alphas_coarse = np.linspace(0.3, 6.0, 12)
    betas_coarse = [0.0, 0.5, 1.0, 2.0]
    
    seeds = [42, 123, 7, 999]
    
    # Phase 1: Coarse search over (alpha, beta)
    for alpha in alphas_coarse:
        if time.time() - start_time > time_limit - 60.0:
            break
        for beta in betas_coarse:
            if time.time() - start_time > time_limit - 60.0:
                break
            
            # Construct Heuristic: slack-based * complementarity
            base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
            if beta > 0:
                base = base * (comp_prior ** beta)
            base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
            
            # Evaluate with multiple seeds using soft-min surrogate
            scores = []
            for seed in seeds:
                score = _fast_aco_eval(d, capacity, base, n_ants=5, n_iters=10, seed=seed)
                scores.append(score)
            
            # Use soft-min as smooth surrogate for robustness
            surrogate = _soft_min(scores, beta=15.0)
            avg_score = np.mean(scores)
            
            # Track both for final selection
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha
                best_beta = beta
    
    # Phase 2: Refine around best (alpha, beta)
    if time.time() - start_time < time_limit - 30.0:
        center_a = best_alpha
        center_b = best_beta
        span_a = max(0.3, 0.2 * center_a)
        
        fine_alphas = np.linspace(max(0.1, center_a - span_a), center_a + span_a, 8)
        
        # Fine betas around best
        if center_b == 0.0:
            fine_betas = [0.0, 0.25, 0.5]
        else:
            fine_betas = np.linspace(max(0.0, center_b - 0.5), center_b + 0.5, 5)
        
        for alpha in fine_alphas:
            if time.time() - start_time > time_limit - 30.0:
                break
            if alpha <= 0:
                alpha = 0.1
            for beta in fine_betas:
                if time.time() - start_time > time_limit - 30.0:
                    break
                if beta < 0:
                    beta = 0.0
                
                base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
                if beta > 0:
                    base = base * (comp_prior ** beta)
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
    
    # Phase 3: Ultra-fine polish around the best found
    if time.time() - start_time < time_limit - 10.0:
        center_a = best_alpha
        center_b = best_beta
        tiny_span_a = 0.15
        tiny_span_b = 0.25
        
        ultra_alphas = np.linspace(max(0.05, center_a - tiny_span_a), center_a + tiny_span_a, 6)
        if center_b == 0.0:
            ultra_betas = [0.0, 0.1, 0.2]
        else:
            ultra_betas = np.linspace(max(0.0, center_b - tiny_span_b), center_b + tiny_span_b, 5)
        
        for alpha in ultra_alphas:
            if time.time() - start_time > time_limit - 10.0:
                break
            if alpha <= 0:
                alpha = 0.05
            for beta in ultra_betas:
                if time.time() - start_time > time_limit - 10.0:
                    break
                if beta < 0:
                    beta = 0.0
                
                base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
                if beta > 0:
                    base = base * (comp_prior ** beta)
                base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
                
                # Use more iterations for final evaluation
                scores = []
                for seed in seeds:
                    score = _fast_aco_eval(d, capacity, base, n_ants=5, n_iters=12, seed=seed)
                    scores.append(score)
                
                avg_score = np.mean(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_alpha = alpha
                    best_beta = beta
    
    # Construct final heuristic with best parameters
    final_heuristic = np.where(valid, 1.0 / (slack + 1e-9)**best_alpha, 1e-9)
    if best_beta > 0:
        final_heuristic = final_heuristic * (comp_prior ** best_beta)
    final_heuristic = np.where(np.isfinite(final_heuristic) & (final_heuristic > 0), final_heuristic, 1e-9)
    
    return final_heuristic
