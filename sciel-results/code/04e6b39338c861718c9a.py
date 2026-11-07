"""Program for ant colony optimization of offline bin packing with heuristic design."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 20.0  # seconds before the 330s kill limit

def _time_budget_remaining():
    return (330.0 - _DEADLINE_MARGIN) - (time.time() - _START_TIME)

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

def _build_complementarity_heuristic(d, capacity, sharpness=5.0, base_power=2.0, tight_bonus=2.0, tight_threshold=5.0):
    """
    Build heuristic based on complementarity score using two-pointer sweep.
    
    Strategy:
    1. Sort demands and for each item, find its "ideal partner" via two-pointer
       (the item that makes the pair sum closest to capacity without exceeding it)
    2. Assign high attractiveness to pairs near these ideal complements
    3. Penalize large deviations from ideal pairing structure
    4. Add tight-fit bonus for pairs with very small waste
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    
    # Sort demands and track original indices
    sorted_indices = np.argsort(d)
    sorted_d = d[sorted_indices]
    
    # For each sorted position, find ideal partner using two-pointer
    # ideal_partner[i] = index j such that d[i]+d[j] is closest to capacity (<= capacity)
    ideal_gap = np.full(n, capacity, dtype=np.float64)
    
    # Two-pointer sweep: for each i, find j that minimizes gap = capacity - d[i] - d[j]
    # where d[i] + d[j] <= capacity
    j = n - 1
    for i in range(n):
        # Move j left while d[i] + d[j] > capacity
        while j > i and sorted_d[i] + sorted_d[j] > capacity:
            j -= 1
        if j >= i:
            gap = capacity - sorted_d[i] - sorted_d[j]
            if gap < ideal_gap[sorted_indices[i]]:
                ideal_gap[sorted_indices[i]] = gap
        # Also check if j-1 gives better gap (for asymmetric cases)
        if j > i + 1:
            gap2 = capacity - sorted_d[i] - sorted_d[j-1]
            if gap2 >= 0 and gap2 < ideal_gap[sorted_indices[i]]:
                ideal_gap[sorted_indices[i]] = gap2
        # Check j+1 if it fits
        if j < n - 1 and sorted_d[i] + sorted_d[j+1] <= capacity:
            gap3 = capacity - sorted_d[i] - sorted_d[j+1]
            if gap3 < ideal_gap[sorted_indices[i]]:
                ideal_gap[sorted_indices[i]] = gap3
    
    # Build pair_sums matrix
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity
    gaps = capacity - pair_sums
    fills = pair_sums / capacity
    
    # For each pair (i,j), compute how close it is to the ideal complement for i
    # ideal_gap[i] is the minimum gap achievable for item i
    # If gap[i,j] is close to ideal_gap[i], it's a good complement
    # We want high attractiveness when gap[i,j] is small AND close to ideal
    
    # Complementarity score: how well j complements i
    # Use the ideal gap of i as reference
    # If gap[i,j] == ideal_gap[i], it's a perfect complement
    # Score = exp(-sharpness * |gap[i,j] - ideal_gap[i]| / capacity)
    
    # But we also want to reward small absolute gaps (tight fit)
    # Combined: base_score * tight_bonus
    
    # Base score: closeness to ideal complement
    # For pair (i,j), the deviation from ideal for i is |gap[i,j] - ideal_gap[i]|
    # But this is asymmetric: j complementing i vs i complementing j
    # The heuristic H[i,j] should reflect placing j after i, so focus on i's perspective
    
    deviation = np.abs(gaps - ideal_gap[:, None])
    base_score = np.exp(-sharpness * deviation / capacity)
    
    # Tight fit bonus: reward pairs with small absolute gap
    tight_mask = gaps <= tight_threshold
    tight_factor = np.where(tight_mask, 1.0 + tight_bonus * (1.0 - gaps / tight_threshold), 1.0)
    
    # Fill-based bonus: prefer fuller bins
    fill_bonus = np.power(np.maximum(fills, 1e-6), base_power)
    
    h = base_score * tight_factor * fill_bonus
    
    # For non-fitting pairs, set to minimum
    h = np.where(fits, h, 1e-9)
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h

def _build_geometric_heuristic(d, capacity, alpha, beta, gamma, delta):
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
    
    # Term 3: bonus for balanced pairs
    term3 = 1.0 + gamma * (1.0 - fills)**2
    
    # Term 4: diversity in sizes might help packing
    diff = np.abs(d[:, None] - d[None, :]) / capacity
    term4 = 1.0 + delta * diff
    
    h = term1 * term2 * term3 * term4
    
    # For non-fitting pairs, set to minimum
    h = np.where(fits, h, 1e-9)
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    
    Strategy: Search over both complementarity and geometric heuristic families,
    using greedy simulation as proxy, then validate top candidates with full ACO.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    best_score = float('inf')
    best_heuristic = None
    
    rng_search = np.random.RandomState(7)
    
    # Stage 1: Greedy search over both heuristic families
    
    # Complementarity family parameters
    comp_candidates = []
    for sharpness in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        for base_power in [1.0, 1.5, 2.0, 3.0]:
            for tight_bonus in [0.0, 1.0, 2.0, 3.0]:
                for tight_threshold in [3.0, 5.0, 8.0, 12.0]:
                    comp_candidates.append(('comp', sharpness, base_power, tight_bonus, tight_threshold))
    
    # Geometric family parameters
    geo_candidates = []
    for alpha in [1.0, 2.0, 3.0, 4.0, 5.0]:
        for beta in [0.5, 1.0, 2.0, 3.0]:
            for gamma in [0.0, 0.5, 1.0, 1.5]:
                for delta in [0.0, 0.5, 1.0]:
                    geo_candidates.append(('geo', alpha, beta, gamma, delta))
    
    # Combine and sample
    all_candidates = comp_candidates + geo_candidates
    max_candidates = 200
    if len(all_candidates) > max_candidates:
        indices = rng_search.choice(len(all_candidates), size=max_candidates, replace=False)
        all_candidates = [all_candidates[i] for i in indices]
    
    rng_search.shuffle(all_candidates)
    
    proxy_results = []
    
    for i, cand in enumerate(all_candidates):
        if _time_budget_remaining() < 30.0:
            break
        
        if cand[0] == 'comp':
            _, sharpness, base_power, tight_bonus, tight_threshold = cand
            h = _build_complementarity_heuristic(d, capacity, sharpness, base_power, tight_bonus, tight_threshold)
        else:
            _, alpha, beta, gamma, delta = cand
            h = _build_geometric_heuristic(d, capacity, alpha, beta, gamma, delta)
        
        score = _greedy_score(d, capacity, h, n_ants=8, seed=42)
        proxy_results.append((score, cand, h))
        
    if not proxy_results:
        # Fallback
        h = _build_complementarity_heuristic(d, capacity, 5.0, 2.0, 2.0, 5.0)
        return h
    
    # Sort by greedy score
    proxy_results.sort(key=lambda x: x[0])
    
    # Stage 2: Validate top candidates with ACO
    top_k = min(12, len(proxy_results))
    for score, cand, h in proxy_results[:top_k]:
        if _time_budget_remaining() < 15.0:
            break
        
        aco_score = _aco_score(d, capacity, h, n_ants=20, n_iterations=30, seed=42)
        
        if aco_score < best_score:
            best_score = aco_score
            best_heuristic = h.copy()
            
    # If no ACO validation was done, use best greedy
    if best_heuristic is None:
        best_heuristic = proxy_results[0][2]
        
    # Final safety check
    if not np.all(np.isfinite(best_heuristic)):
        best_heuristic = np.where(np.isfinite(best_heuristic), best_heuristic, 1e-9)
    if not np.all(best_heuristic > 0):
        best_heuristic = np.where(best_heuristic > 0, best_heuristic, 1e-9)
        
    return best_heuristic
# EVOLVE-BLOCK-END
