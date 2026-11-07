"""Program for ant colony optimization of offline bin packing with heuristic design."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 20.0  # seconds before the 330s kill limit

def _time_budget_remaining():
    return (330.0 - _DEADLINE_MARGIN) - (time.time() - _START_TIME)

def _build_complementarity_heuristic(d, capacity, alpha, beta, gamma):
    """
    Complementarity-based heuristic: identify near-optimal partner pairs using
    two-pointer sweep on sorted demands, then build heuristic that favors these pairings.
    
    H_ij = base_heuristic * (1 + gamma * complementarity_score(i,j))
    where complementarity_score is high if j is a good partner for i in a tight bin.
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity
    gaps = capacity - pair_sums
    fills = pair_sums / capacity
    
    # Base terms
    term1 = np.power(np.maximum(fills, 1e-6), alpha)
    term2 = np.power(1.0 / (1.0 + gaps), beta)
    
    # Complementarity: for each item i, find its "ideal partner" j such that
    # d_i + d_j is as close to capacity as possible (tight fit)
    # Use two-pointer on sorted array
    sorted_idx = np.argsort(d)
    sorted_d = d[sorted_idx]
    
    # For each position in sorted array, find best partner
    complementarity = np.zeros((n, n))
    
    # Two-pointer approach: for each i, the best j is the largest item that fits with i
    # We'll compute for sorted positions and map back
    best_partner_score = np.zeros(n)
    best_partner_idx = np.zeros(n, dtype=int)
    
    left = 0
    right = n - 1
    
    # For each item (in sorted order), find the item that makes the tightest fit
    for pos in range(n):
        # Find j such that sorted_d[pos] + sorted_d[j] <= capacity and is maximized
        # Binary search for the largest j where sorted_d[j] <= capacity - sorted_d[pos]
        remaining = capacity - sorted_d[pos]
        # Find rightmost index where sorted_d <= remaining
        idx = np.searchsorted(sorted_d, remaining, side='right') - 1
        if idx >= 0 and idx < n:
            best_partner_score[pos] = 1.0 / (1.0 + (capacity - sorted_d[pos] - sorted_d[idx]))
            best_partner_idx[pos] = idx
    
    # Map back to original indices
    for pos in range(n):
        orig_i = sorted_idx[pos]
        orig_j = sorted_idx[best_partner_idx[pos]]
        # Symmetric: both i->j and j->i get the complementarity score
        complementarity[orig_i, orig_j] = best_partner_score[pos]
        complementarity[orig_j, orig_i] = best_partner_score[pos]
    
    # Also consider secondary partners (next best fit)
    for pos in range(n):
        remaining = capacity - sorted_d[pos]
        idx = np.searchsorted(sorted_d, remaining, side='right') - 1
        if idx >= 1 and idx < n:
            second_idx = idx - 1
            second_score = 0.5 / (1.0 + (capacity - sorted_d[pos] - sorted_d[second_idx]))
            orig_i = sorted_idx[pos]
            orig_j = sorted_idx[second_idx]
            complementarity[orig_i, orig_j] = max(complementarity[orig_i, orig_j], second_score)
            complementarity[orig_j, orig_i] = max(complementarity[orig_j, orig_i], second_score)
    
    # Combine: base * (1 + gamma * complementarity)
    h = term1 * term2 * (1.0 + gamma * complementarity)
    
    # For non-fitting pairs, set to minimum
    h = np.where(fits, h, 1e-9)
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h

def _greedy_score(d, capacity, heuristic, n_ants=10, seed=42):
    """
    Fast greedy simulation: pick item j that maximizes heuristic[current, j]
    among fitting items. This approximates the ACO when pheromone is uniform.
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.RandomState(seed)
    
    best_bins = float('inf')
    
    for _ in range(n_ants):
        placed = np.zeros(n, dtype=bool)
        bins_used = 0
        current_bin_sum = 0.0
        last_item = -1
        
        # Open first bin, random start
        idx = rng.randint(n)
        placed[idx] = True
        current_bin_sum = d[idx]
        last_item = idx
        bins_used = 1
        
        while not np.all(placed):
            remaining_space = capacity - current_bin_sum
            fits_mask = (~placed) & (d <= remaining_space)
            
            if not np.any(fits_mask):
                # Open new bin
                bins_used += 1
                available = np.where(~placed)[0]
                idx = available[rng.randint(len(available))]
                placed[idx] = True
                current_bin_sum = d[idx]
                last_item = idx
                continue
            
            # Greedy choice: max heuristic[last_item, j] among fitting j
            row_he = heuristic[last_item, :]
            scores = np.where(fits_mask, row_he, 0.0)
            
            idx = np.argmax(scores)
            
            if scores[idx] == 0:
                candidates = np.where(fits_mask)[0]
                idx = candidates[rng.randint(len(candidates))]
            
            placed[idx] = True
            current_bin_sum += d[idx]
            last_item = idx
            
        if bins_used < best_bins:
            best_bins = bins_used
            
    return best_bins

def _aco_score(d, capacity, heuristic, n_ants=20, n_iterations=30, seed=42):
    """
    Full ACO simulation as per task description.
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    
    pheromone = np.ones((n, n))
    rng = np.random.RandomState(seed)
    
    best_bins_ever = float('inf')
    
    for _ in range(n_iterations):
        best_bins_this_iter = float('inf')
        
        for _ in range(n_ants):
            placed = np.zeros(n, dtype=bool)
            bins_used = 0
            current_bin_sum = 0.0
            last_item = -1
            
            # Open a bin, place a random fitting item
            idx = rng.randint(n)
            placed[idx] = True
            current_bin_sum = d[idx]
            last_item = idx
            bins_used = 1
            
            while not np.all(placed):
                remaining_space = capacity - current_bin_sum
                fits_mask = (~placed) & (d <= remaining_space)
                
                if not np.any(fits_mask):
                    bins_used += 1
                    available = np.where(~placed)[0]
                    idx = available[rng.randint(len(available))]
                    placed[idx] = True
                    current_bin_sum = d[idx]
                    last_item = idx
                    continue
                
                row_ph = pheromone[last_item, :]
                row_he = heuristic[last_item, :]
                
                scores = row_ph * row_he
                scores = np.where(fits_mask, scores, 0.0)
                
                total_score = np.sum(scores)
                if total_score == 0:
                    candidates = np.where(fits_mask)[0]
                    idx = candidates[rng.randint(len(candidates))]
                else:
                    probs = scores / total_score
                    idx = rng.choice(n, p=probs)
                    
                placed[idx] = True
                current_bin_sum += d[idx]
                last_item = idx
                
            if bins_used < best_bins_this_iter:
                best_bins_this_iter = bins_used
                
        if best_bins_this_iter < best_bins_ever:
            best_bins_ever = best_bins_this_iter
            
        # Update pheromone
        pheromone *= 0.9
        pheromone += 1.0 / best_bins_this_iter
        
    return best_bins_ever

def _build_heuristic(d, capacity, alpha, beta, gamma, delta):
    """
    Build heuristic matrix from parameters.
    H_ij = ((d_i+d_j)/C)^alpha * (1/(1+gap))^beta * (1 + gamma*(1 - fill)^2 + delta*|d_i - d_j|/C)
    where fill = (d_i+d_j)/C, gap = C - (d_i+d_j)
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity
    gaps = capacity - pair_sums
    fills = pair_sums / capacity
    
    # Term 1: prefer fuller bins (higher fill)
    term1 = np.power(np.maximum(fills, 1e-6), alpha)
    
    # Term 2: prefer tight fit (small gap)
    term2 = np.power(1.0 / (1.0 + gaps), beta)
    
    # Term 3: bonus for balanced pairs (similar sizes) and penalty for very different
    # Actually, for bin packing, we might want to prefer items that complement
    # Let's use: bonus for making the bin close to full
    # fill^2 encourages packing more tightly
    term3 = 1.0 + gamma * (1.0 - fills)**2
    
    # Term 4: diversity in sizes might help packing
    diff = np.abs(d[:, None] - d[None, :]) / capacity
    term4 = 1.0 + delta * diff
    
    h = term1 * term2 * term3 * term4
    
    # For non-fitting pairs, set to minimum
    h = np.where(fits, h, 1e-9)
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h

def _build_heuristic_v2(d, capacity, alpha, beta, gamma):
    """
    Alternative heuristic: emphasize remaining space utilization.
    H_ij = (remaining_after / C)^alpha * (1/(1+gap))^beta * (1 + gamma*fill)
    where remaining_after = C - d_i - d_j, fill = (d_i+d_j)/C
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity
    gaps = capacity - pair_sums
    fills = pair_sums / capacity
    remaining = gaps / capacity
    
    # Prefer items that leave less space (tighter fit)
    term1 = np.power(np.maximum(1.0 - remaining, 1e-6), alpha)
    
    # Penalize large gaps
    term2 = np.power(1.0 / (1.0 + gaps), beta)
    
    # Bonus for high fill
    term3 = 1.0 + gamma * fills
    
    h = term1 * term2 * term3
    
    h = np.where(fits, h, 1e-9)
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h

def _fast_greedy_score(d, capacity, heuristic, n_ants=6, seed=42):
    """
    Fast greedy simulation with early termination for quick proxy scoring.
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.RandomState(seed)
    
    best_bins = float('inf')
    
    for _ in range(n_ants):
        placed = np.zeros(n, dtype=bool)
        bins_used = 0
        current_bin_sum = 0.0
        last_item = -1
        
        # Open first bin, random start
        idx = rng.randint(n)
        placed[idx] = True
        current_bin_sum = d[idx]
        last_item = idx
        bins_used = 1
        
        while not np.all(placed):
            remaining_space = capacity - current_bin_sum
            fits_mask = (~placed) & (d <= remaining_space)
            
            if not np.any(fits_mask):
                bins_used += 1
                available = np.where(~placed)[0]
                idx = available[rng.randint(len(available))]
                placed[idx] = True
                current_bin_sum = d[idx]
                last_item = idx
                continue
            
            row_he = heuristic[last_item, :]
            scores = np.where(fits_mask, row_he, 0.0)
            
            idx = np.argmax(scores)
            
            if scores[idx] == 0:
                candidates = np.where(fits_mask)[0]
                idx = candidates[rng.randint(len(candidates))]
            
            placed[idx] = True
            current_bin_sum += d[idx]
            last_item = idx
            
        if bins_used < best_bins:
            best_bins = bins_used
            
    return best_bins

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    
    Strategy: Search over both local-waste and complementarity-based heuristics,
    using fast greedy proxy then full ACO validation.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    best_score = float('inf')
    best_heuristic = None
    
    rng_search = np.random.RandomState(7)
    
    # Stage 1: Search local-waste heuristics
    n_samples = 80
    alphas = np.append(rng_search.uniform(0.5, 8.0, n_samples), [1.0, 2.0, 3.0, 4.0, 5.0])
    betas = np.append(rng_search.uniform(0.0, 5.0, n_samples), [0.5, 1.0, 2.0, 3.0])
    gammas = np.append(rng_search.uniform(0.0, 2.0, n_samples), [0.0, 0.5, 1.0, 1.5])
    
    candidates_local = []
    for a in alphas:
        for b in betas:
            for g in gammas:
                candidates_local.append((a, b, g))
    
    max_local = 200
    if len(candidates_local) > max_local:
        indices = rng_search.choice(len(candidates_local), size=max_local, replace=False)
        candidates_local = [candidates_local[i] for i in indices]
    
    rng_search.shuffle(candidates_local)
    
    proxy_results = []
    
    for i, (alpha, beta, gamma) in enumerate(candidates_local):
        if _time_budget_remaining() < 40.0:
            break
        
        h = _build_heuristic(d, capacity, alpha, beta, gamma, delta=0.5)
        score = _fast_greedy_score(d, capacity, h, n_ants=6, seed=42)
        proxy_results.append((score, 'local', alpha, beta, gamma, h))
        
    # Stage 2: Search complementarity-based heuristics
    n_comp = 60
    comp_alphas = np.append(rng_search.uniform(0.5, 6.0, n_comp), [1.0, 2.0, 3.0, 4.0])
    comp_betas = np.append(rng_search.uniform(0.0, 4.0, n_comp), [0.5, 1.0, 2.0, 3.0])
    comp_gammas = np.append(rng_search.uniform(0.0, 3.0, n_comp), [0.5, 1.0, 1.5, 2.0])
    
    candidates_comp = []
    for a in comp_alphas:
        for b in comp_betas:
            for g in comp_gammas:
                candidates_comp.append((a, b, g))
    
    max_comp = 150
    if len(candidates_comp) > max_comp:
        indices = rng_search.choice(len(candidates_comp), size=max_comp, replace=False)
        candidates_comp = [candidates_comp[i] for i in indices]
    
    rng_search.shuffle(candidates_comp)
    
    for i, (alpha, beta, gamma) in enumerate(candidates_comp):
        if _time_budget_remaining() < 40.0:
            break
        
        h = _build_complementarity_heuristic(d, capacity, alpha, beta, gamma)
        score = _fast_greedy_score(d, capacity, h, n_ants=6, seed=42)
        proxy_results.append((score, 'comp', alpha, beta, gamma, h))
        
    if not proxy_results:
        # Fallback
        h = _build_complementarity_heuristic(d, capacity, alpha=2.0, beta=1.0, gamma=1.0)
        return h
    
    # Sort by greedy score
    proxy_results.sort(key=lambda x: x[0])
    
    # Stage 3: Validate top candidates with ACO
    top_k = min(12, len(proxy_results))
    for score, heur_type, alpha, beta, gamma, h in proxy_results[:top_k]:
        if _time_budget_remaining() < 20.0:
            break
        
        aco_score = _aco_score(d, capacity, h, n_ants=20, n_iterations=30, seed=42)
        
        if aco_score < best_score:
            best_score = aco_score
            best_heuristic = h.copy()
            
    if best_heuristic is None:
        best_heuristic = proxy_results[0][5]
        
    # Final safety check
    if not np.all(np.isfinite(best_heuristic)):
        best_heuristic = np.where(np.isfinite(best_heuristic), best_heuristic, 1e-9)
    if not np.all(best_heuristic > 0):
        best_heuristic = np.where(best_heuristic > 0, best_heuristic, 1e-9)
        
    return best_heuristic
# EVOLVE-BLOCK-END
