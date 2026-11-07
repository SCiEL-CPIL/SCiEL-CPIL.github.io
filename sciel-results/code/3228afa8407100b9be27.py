"""Program for ant colony optimization of offline bin packing with heuristic design."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 20.0  # seconds before the 330s kill limit

def _time_budget_remaining():
    return (330.0 - _DEADLINE_MARGIN) - (time.time() - _START_TIME)

def _stochastic_greedy_score(d, capacity, heuristic, n_ants=5, seed=42, temperature=1.0):
    """
    Fast proxy: Simulates ACO-like choice without pheromone updates.
    Picks item j with probability proportional to heuristic[current, j] ^ temperature
    among fitting items.
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
            
            # Get heuristic scores for fitting items
            row_he = heuristic[last_item, :]
            scores = np.where(fits_mask, row_he, 0.0)
            
            # Apply temperature
            scores = np.power(scores, temperature)
            
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

def _build_heuristic(d, capacity, p, q, r):
    """
    H_ij = (fill)^p * (1/(1+gap/10))^q * (1 + r * exp(-gap^2))
    """
    n = len(d)
    d = np.asarray(d, dtype=np.float64)
    
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity
    gaps = capacity - pair_sums
    fills = pair_sums / capacity
    
    # Term 1: Prefer fuller bins
    term1 = np.power(np.maximum(fills, 1e-6), p)
    
    # Term 2: Prefer tight fit (small gap)
    # Using a smooth decay for gap
    term2 = np.power(1.0 / (1.0 + gaps / 10.0), q)
    
    # Term 3: Bonus for very tight fits (gap close to 0)
    # This helps distinguish between "good" and "great" fits
    term3 = 1.0 + r * np.exp(-gaps * gaps)
    
    h = term1 * term2 * term3
    
    # For non-fitting pairs, set to minimum
    h = np.where(fits, h, 1e-9)
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    best_score = float('inf')
    best_heuristic = None
    
    # Define search space for parameters p, q, r
    # p: power of fill ratio (higher = more aggressive preference for full bins)
    # q: power of gap penalty (higher = more aggressive penalty for waste)
    # r: weight of tight-fit bonus
    
    # We will sample from a grid + random perturbations
    p_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    q_vals = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    r_vals = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    candidates = []
    for p in p_vals:
        for q in q_vals:
            for r in r_vals:
                candidates.append((p, q, r))
    
    # Add some random variations to escape local optima
    rng_search = np.random.RandomState(123)
    for _ in range(50):
        p = rng_search.uniform(0.5, 10.0)
        q = rng_search.uniform(0.5, 5.0)
        r = rng_search.uniform(0.0, 5.0)
        candidates.append((p, q, r))
    
    # Evaluate candidates with stochastic greedy proxy
    proxy_results = []
    
    # Limit number of proxies to fit in time budget
    # Each proxy is ~5 ants * 120 items * O(N) ops. Should be fast.
    # 200 candidates * 5 ants * 120 steps * 120 items check ~ 14M ops. Fast.
    
    for p, q, r in candidates:
        if _time_budget_remaining() < 30.0:
            break
            
        h = _build_heuristic(d, capacity, p, q, r)
        
        # Use temperature 1.0 for proxy (standard stochastic greedy)
        score = _stochastic_greedy_score(d, capacity, h, n_ants=5, seed=42, temperature=1.0)
        proxy_results.append((score, p, q, r, h))
        
    if not proxy_results:
        # Fallback
        h = _build_heuristic(d, capacity, 3.0, 2.0, 1.0)
        return h
        
    # Sort by proxy score (lower is better)
    proxy_results.sort(key=lambda x: x[0])
    
    # Validate top candidates with full ACO
    # Top K = 5 to 10 depending on time
    top_k = min(10, len(proxy_results))
    
    for i in range(top_k):
        if _time_budget_remaining() < 20.0:
            break
            
        score_proxy, p, q, r, h = proxy_results[i]
        
        # Run full ACO
        aco_score = _aco_score(d, capacity, h, n_ants=20, n_iterations=30, seed=42)
        
        if aco_score < best_score:
            best_score = aco_score
            best_heuristic = h.copy()
            
    # If no ACO was run or all failed, use best proxy
    if best_heuristic is None:
        best_heuristic = proxy_results[0][4]
        
    # Final safety check
    if not np.all(np.isfinite(best_heuristic)):
        best_heuristic = np.where(np.isfinite(best_heuristic), best_heuristic, 1e-9)
    if not np.all(best_heuristic > 0):
        best_heuristic = np.where(best_heuristic > 0, best_heuristic, 1e-9)
        
    return best_heuristic
# EVOLVE-BLOCK-END
