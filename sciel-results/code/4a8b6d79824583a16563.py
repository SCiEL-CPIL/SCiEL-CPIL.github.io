<<<<<<< SEARCH
import numpy as np
import time

def _run_aco_fast(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Fast ACO simulation using precomputed structures.
    Returns the minimum number of bins used by any ant.
    """
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    # Precompute which items can follow which (based on capacity)
    # can_follow[i, j] = True if d[i] + d[j] <= capacity
    can_follow = (d[:, None] + d[None, :]) <= capacity
    
    # Precompute cumulative probability arrays for faster sampling
    # We'll update these as pheromone changes
    
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
            
            # Open first bin - pick random item
            j = int(rng.integers(0, n))
            used[j] = True
            current_fill = d[j]
            current_item = j
            
            # Place remaining items
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
                    j = int(unused_indices[rng.integers(0, len(unused_indices))])
                    used[j] = True
                    current_fill = d[j]
                    current_item = j
                    continue
                else:
                    # Get probabilities for candidates only
                    row_tau = tau[current_item, :]
                    row_heur = heuristic[current_item, :]
                    probs = row_tau * row_heur
                    probs[~mask] = 0.0
                    
                    total_prob = np.sum(probs)
                    if total_prob <= 0:
                        candidates = np.where(mask)[0]
                        j = int(candidates[rng.integers(0, len(candidates))])
                    else:
                        probs = probs / total_prob
                        cum_probs = np.cumsum(probs)
                        r = rng.random()
                        idx = int(np.searchsorted(cum_probs, r))
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
=======
import numpy as np
import time

def _run_aco_fast(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Fast ACO simulation using precomputed structures.
    Returns the minimum number of bins used by any ant.
    """
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    # Precompute which items fit in remaining space after placing item i
    # fit_after[i] = list of items j such that d[i] + d[j] <= capacity
    fit_after = []
    for i in range(n):
        remaining = capacity - d[i]
        candidates = np.where(d <= remaining)[0]
        fit_after.append(candidates)
    
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
            
            # Open first bin - pick random item
            j = int(rng.integers(0, n))
            used[j] = True
            current_fill = d[j]
            current_item = j
            
            # Place remaining items
            while True:
                remaining = capacity - current_fill
                
                # Candidates: not used and d[j] <= remaining
                # Use precomputed fit_after for speed
                candidates_all = fit_after[current_item]
                # Filter to unused items
                candidate_mask = ~used[candidates_all]
                candidates = candidates_all[candidate_mask]
                
                if len(candidates) == 0:
                    # Open new bin
                    bins += 1
                    unused_indices = np.where(~used)[0]
                    if len(unused_indices) == 0:
                        break
                    j = int(unused_indices[rng.integers(0, len(unused_indices))])
                    used[j] = True
                    current_fill = d[j]
                    current_item = j
                    continue
                else:
                    # Get probabilities for candidates only
                    row_tau = tau[current_item, candidates]
                    row_heur = heuristic[current_item, candidates]
                    probs = row_tau * row_heur
                    
                    total_prob = np.sum(probs)
                    if total_prob <= 0:
                        j = int(candidates[rng.integers(0, len(candidates))])
                    else:
                        probs = probs / total_prob
                        cum_probs = np.cumsum(probs)
                        r = rng.random()
                        idx = int(np.searchsorted(cum_probs, r))
                        if idx >= len(candidates):
                            idx = len(candidates) - 1
                        j = int(candidates[idx])
                        
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
>>>>>>> REPLACE

