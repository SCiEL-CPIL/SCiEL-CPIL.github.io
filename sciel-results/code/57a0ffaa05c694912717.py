import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0

def _time_budget_remaining():
    return (330.0 - _DEADLINE_MARGIN) - (time.time() - _START_TIME)

def _greedy_ffd(demands, capacity):
    """
    First Fit Decreasing: sort items descending, place each in first bin that fits.
    Returns list of bins, each bin is a list of item indices.
    """
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    order = np.argsort(-d)  # descending
    bins = []
    bin_sums = []
    item_to_bin = {}
    
    for idx in order:
        placed = False
        for b_idx in range(len(bins)):
            if bin_sums[b_idx] + d[idx] <= capacity:
                bins[b_idx].append(idx)
                bin_sums[b_idx] += d[idx]
                item_to_bin[idx] = b_idx
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_sums.append(d[idx])
            item_to_bin[idx] = len(bins) - 1
    
    return bins

def _simulate_aco_fast(demands, capacity, heuristic, n_ants=20, n_iterations=30, seed=42):
    """
    Fast ACO simulation to evaluate a heuristic.
    Returns fewest bins used by any ant in the final iteration.
    """
    np.random.seed(seed)
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    pheromone = np.ones((n, n))
    best_bins_ever = float('inf')
    
    for _ in range(n_iterations):
        best_bins_this_iter = float('inf')
        
        for _ in range(n_ants):
            placed = np.zeros(n, dtype=bool)
            bins_used = 0
            current_bin_sum = 0.0
            last_item = -1
            
            # Open first bin, place random fitting item
            idx = np.random.randint(n)
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
                    idx = available[np.random.randint(len(available))]
                    placed[idx] = True
                    current_bin_sum = d[idx]
                    last_item = idx
                    continue
                
                scores = pheromone[last_item, :] * heuristic[last_item, :]
                scores = np.where(fits_mask, scores, 0.0)
                total_score = np.sum(scores)
                
                if total_score == 0:
                    candidates = np.where(fits_mask)[0]
                    idx = candidates[np.random.randint(len(candidates))]
                else:
                    probs = scores / total_score
                    idx = np.random.choice(n, p=probs)
                
                placed[idx] = True
                current_bin_sum += d[idx]
                last_item = idx
        
        if bins_used < best_bins_this_iter:
            best_bins_this_iter = bins_used
        
        if best_bins_this_iter < best_bins_ever:
            best_bins_ever = best_bins_this_iter
        
        pheromone *= 0.9
        pheromone += 1.0 / best_bins_this_iter
    
    return best_bins_ever

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    
    Strategy: Data-driven empirical pair statistics.
    1. Run FFD to get a high-quality solution.
    2. Count how often each pair (i, j) appears in the same bin in the FFD solution.
    3. Use these counts as the basis for the heuristic, with a smooth prior to avoid zeros.
    4. Validate with a quick ACO simulation and return the best variant.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Step 1: Get FFD solution
    bins = _greedy_ffd(d, capacity)
    
    # Step 2: Build empirical pair matrix
    # Count co-occurrence in same bin
    pair_count = np.zeros((n, n))
    for bin_items in bins:
        for i_idx in bin_items:
            for j_idx in bin_items:
                if i_idx != j_idx:
                    pair_count[i_idx, j_idx] += 1
    
    # Step 3: Create heuristic from pair counts
    # We want to reward pairs that are likely to fit together well.
    # Normalize and add a small epsilon to avoid zeros.
    # Also, we should only reward pairs that actually fit.
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity
    
    # Empirical heuristic: count-based
    # H[i,j] = pair_count[i,j] + epsilon, if fits, else 1e-9
    epsilon = 1e-6
    h_empirical = np.where(fits, pair_count + epsilon, 1e-9)
    
    # Also try a normalized version
    # H[i,j] = (pair_count[i,j] + epsilon) / (max_count + epsilon), if fits
    max_count = np.max(pair_count) if n > 0 else 1
    h_normalized = np.where(fits, (pair_count + epsilon) / (max_count + epsilon), 1e-9)
    
    # Also try a hybrid: empirical * fill_ratio
    fill_ratio = np.where(fits, pair_sums / capacity, 0)
    h_hybrid = np.where(fits, (pair_count + epsilon) * fill_ratio, 1e-9)
    
    # Ensure all are positive and finite
    candidates = [
        ('empirical', h_empirical),
        ('normalized', h_normalized),
        ('hybrid', h_hybrid)
    ]
    
    # Also add the seed-like heuristic as a fallback
    h_seed = np.where(fits, fill_ratio, 1e-9)
    candidates.append(('seed', h_seed))
    
    # Validate each candidate with a quick ACO simulation
    best_score = float('inf')
    best_heur = None
    
    for name, h in candidates:
        if _time_budget_remaining() < 5.0:
            break
        
        # Ensure h is valid
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        
        # Quick evaluation with one seed
        score = _simulate_aco_fast(d, capacity, h, n_ants=10, n_iterations=10, seed=42)
        
        if score < best_score:
            best_score = score
            best_heur = h.copy()
    
    # If no heuristic was evaluated or best is None, return seed-like
    if best_heur is None:
        best_heur = h_seed
        best_heur = np.where(np.isfinite(best_heur) & (best_heur > 0), best_heur, 1e-9)
    
    return best_heur
