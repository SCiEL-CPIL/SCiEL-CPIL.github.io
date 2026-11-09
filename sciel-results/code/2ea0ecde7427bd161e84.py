import numpy as np
import time

def _run_aco_fast(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Fast ACO simulation. Returns minimum bins used by any ant over all iterations.
    """
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    tau = np.ones((n, n))
    best_overall = np.inf
    rng = np.random.default_rng(seed)
    
    for it in range(n_iters):
        min_bins_this_iter = n + 1  # worst case
        
        for ant in range(n_ants):
            used = np.zeros(n, dtype=bool)
            bins = 1
            current_fill = 0.0
            current_item = -1
            
            # Place first item: random
            j = rng.integers(0, n)
            used[j] = True
            current_fill = d[j]
            current_item = j
            
            while True:
                remaining = capacity - current_fill
                # Candidates: not used, fits
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
                
                # Compute probabilities for candidates
                row_tau = tau[current_item, :]
                row_heur = heuristic[current_item, :]
                probs = row_tau * row_heur
                probs[~mask] = 0.0
                
                total = probs.sum()
                if total <= 0:
                    candidates = np.where(mask)[0]
                    j = candidates[rng.integers(0, len(candidates))]
                else:
                    # Use multinomial sampling
                    j = rng.choice(n, p=probs / total)
                
                used[j] = True
                current_fill += d[j]
                current_item = j
            
        min_bins_this_iter = bins if bins < min_bins_this_iter else min_bins_this_iter
        if min_bins_this_iter < best_overall:
            best_overall = min_bins_this_iter
        
        # Pheromone update
        tau = tau * 0.9 + (1.0 / min_bins_this_iter)
    
    return best_overall

def heuristics(demands, capacity):
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    start_time = time.time()
    time_limit = 310.0  # Safety margin before 330s kill
    
    # Pre-compute pair sums and validity
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    slack = np.where(valid, capacity - pair_sum, 1e9)
    
    best_score = np.inf
    best_heuristic = None
    best_params = None
    
    def _make_heuristic(alpha, beta, target_frac):
        """Create a combined heuristic: 
        - alpha: best-fit weight (1/slack^alpha)
        - beta: worst-fit weight (slack^beta)  
        - target_frac: distance-to-target weight
        """
        h = np.ones((n, n))
        
        # Best-fit component
        if alpha > 0:
            bf = np.where(valid, 1.0 / (slack + 1e-6)**alpha, 0.0)
            h *= bf
        
        # Worst-fit component
        if beta > 0:
            wf = np.where(valid, (slack + 1e-6)**beta, 0.0)
            h *= wf
        
        # Target distance component
        if target_frac > 0:
            target = capacity * target_frac
            dist = np.abs(pair_sum - target)
            td = np.where(valid, 1.0 / (dist + 1e-6), 0.0)
            h *= td
        
        # Zero out invalid pairs
        h[~valid] = 1e-9
        
        # Ensure positive and finite
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        
        return h
    
    def _evaluate_heuristic(h, seeds=(42, 123, 456)):
        scores = []
        for s in seeds:
            score = _run_aco_fast(d, capacity, h, n_ants=20, n_iters=30, seed=s)
            scores.append(score)
        return np.mean(scores)
    
    # Phase 1: Coarse grid search over alpha, beta, target_frac
    alpha_vals = [0, 0.25, 0.5, 1.0, 1.5, 2.0]
    beta_vals = [0, 0.5, 1.0, 1.5]
    target_frac_vals = [0, 0.5, 0.6, 0.7]
    
    for alpha in alpha_vals:
        for beta in beta_vals:
            for tf in target_frac_vals:
                if time.time() - start_time > time_limit:
                    break
                
                # Skip if all are zero (degenerate)
                if alpha == 0 and beta == 0 and tf == 0:
                    continue
                
                h = _make_heuristic(alpha, beta, tf)
                score = _evaluate_heuristic(h)
                
                if score < best_score:
                    best_score = score
                    best_heuristic = h.copy()
                    best_params = (alpha, beta, tf)
    
    # Phase 2: Fine-tune around best parameters
    if best_params is not None and time.time() - start_time < time_limit:
        alpha0, beta0, tf0 = best_params
        
        # Local search in parameter space
        for da in [-0.25, 0.25]:
            for db in [-0.25, 0.25]:
                for dtf in [-0.05, 0.05]:
                    if time.time() - start_time > time_limit:
                        break
                    
                    alpha = max(0, alpha0 + da)
                    beta = max(0, beta0 + db)
                    tf = max(0, min(1.0, tf0 + dtf))
                    
                    if alpha == 0 and beta == 0 and tf == 0:
                        continue
                    
                    h = _make_heuristic(alpha, beta, tf)
                    score = _evaluate_heuristic(h)
                    
                    if score < best_score:
                        best_score = score
                        best_heuristic = h.copy()
                        best_params = (alpha, beta, tf)
    
    # Phase 3: Smooth surrogate refinement
    # Use a soft-min surrogate to smooth the discrete bin-count objective
    # and do a final polish with more iterations
    if best_params is not None and time.time() - start_time < time_limit - 30:
        alpha0, beta0, tf0 = best_params
        
        # Try a few more refined points with higher precision
        for da in [-0.1, 0.1]:
            for db in [-0.1, 0.1]:
                for dtf in [-0.02, 0.02]:
                    if time.time() - start_time > time_limit:
                        break
                    
                    alpha = max(0, alpha0 + da)
                    beta = max(0, beta0 + db)
                    tf = max(0, min(1.0, tf0 + dtf))
                    
                    if alpha == 0 and beta == 0 and tf == 0:
                        continue
                    
                    h = _make_heuristic(alpha, beta, tf)
                    score = _evaluate_heuristic(h, seeds=(42, 123, 456, 789, 101))
                    
                    if score < best_score:
                        best_score = score
                        best_heuristic = h.copy()
                        best_params = (alpha, beta, tf)
    
    if best_heuristic is None:
        # Fallback: simple best-fit
        base = np.where(valid, 1.0 / (slack + 1e-6), 1e-9)
        best_heuristic = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
    
    return best_heuristic
