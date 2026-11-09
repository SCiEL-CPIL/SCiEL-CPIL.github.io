<<<<<<< SEARCH
import numpy as np
import time

def _run_aco_fast(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Fast ACO simulation using precomputed candidate lists and vectorized operations.
    Returns the minimum number of bins used by any ant.
    """
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    # Precompute: for each item i, which items j can follow i (d[i] + d[j] <= capacity)
    # This is the "valid" mask
    pair_sum = d[:, None] + d[None, :]
    valid_mask = pair_sum <= capacity
    
    # Precompute candidate indices for each item (items that can follow it)
    # For efficiency, we'll use numpy arrays
    candidates_list = [np.where(valid_mask[i])[0] for i in range(n)]
    candidate_counts = np.array([len(c) for c in candidates_list])
    
    # Pheromone initialization
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
            
            # Open first bin - place random item
            j = rng.integers(0, n)
            used[j] = True
            current_fill = d[j]
            current_item = j
            
            # Place remaining items
            while True:
                remaining = capacity - current_fill
                # Get candidates that are unused and fit
                cands = candidates_list[current_item]
                if len(cands) == 0:
                    break
                # Filter to unused items that fit
                valid_cands = cands[~used[cands]]
                if len(valid_cands) == 0:
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
                
                # Compute probabilities for valid candidates
                row_tau = tau[current_item, valid_cands]
                row_heur = heuristic[current_item, valid_cands]
                probs = row_tau * row_heur
                
                total_prob = np.sum(probs)
                if total_prob <= 0:
                    j = valid_cands[rng.integers(0, len(valid_cands))]
                else:
                    probs = probs / total_prob
                    r = rng.random()
                    # Use searchsorted on cumulative sum
                    cum_probs = np.cumsum(probs)
                    idx = np.searchsorted(cum_probs, r)
                    if idx >= len(valid_cands):
                        idx = len(valid_cands) - 1
                    j = valid_cands[idx]
                
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
    Fast ACO simulation using precomputed candidate lists and vectorized operations.
    Returns the minimum number of bins used by any ant.
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
        ant_bins = []
        
        for ant in range(n_ants):
            used = np.zeros(n, dtype=bool)
            bins = 1
            current_fill = 0.0
            
            # Open first bin - place random item
            j = rng.integers(0, n)
            used[j] = True
            current_fill = d[j]
            current_item = j
            
            # Place remaining items
            while True:
                cands = candidates_list[current_item]
                if len(cands) == 0:
                    break
                valid_cands = cands[~used[cands]]
                if len(valid_cands) == 0:
                    bins += 1
                    unused_indices = np.where(~used)[0]
                    if len(unused_indices) == 0:
                        break
                    j = unused_indices[rng.integers(0, len(unused_indices))]
                    used[j] = True
                    current_fill = d[j]
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
                current_fill += d[j]
                current_item = j

            ant_bins.append(bins)
            
        min_bins = min(ant_bins)
        if min_bins < best_overall:
            best_overall = min_bins
            
        tau = tau * 0.9 + (1.0 / min_bins)
        
    return best_overall
>>>>>>> REPLACE

