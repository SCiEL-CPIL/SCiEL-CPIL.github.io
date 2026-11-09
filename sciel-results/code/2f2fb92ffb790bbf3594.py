"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Return an item-by-item matrix of attractiveness for placing item j right
    after item i in the same bin. Larger is more attractive.
    
    Uses a combination of:
    - Slack-based reward: prefer tight fits (small slack after placing both)
    - Size complementarity: prefer pairs that together fill more of the bin
    - Avoid very small items paired with very large ones if they waste space
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Pair sums
    pair = d[:, None] + d[None, :]
    
    # Mask for pairs that fit
    mask = pair <= capacity
    
    # For fitting pairs, compute slack = capacity - pair_sum
    slack = capacity - pair  # positive where mask is True
    
    # Heuristic: reward tight fits strongly
    # Use 1/slack^alpha to strongly favor small slack
    # alpha > 1 amplifies the preference for very tight fits
    alpha = 2.0
    
    # Avoid division by zero: where slack is 0 (perfect fit), use a large value
    # where slack is very small, use large values
    # We'll use a smooth function: 1 / (slack + epsilon) ** alpha
    # But we want to cap it to avoid extreme values
    
    eps = 1e-6
    heuristic_fitting = 1.0 / (slack + eps) ** alpha
    
    # Normalize to a reasonable range
    # Find the max value among fitting pairs
    fitting_vals = heuristic_fitting[mask]
    if len(fitting_vals) > 0:
        max_val = np.max(fitting_vals)
        if max_val > 0:
            heuristic_fitting = heuristic_fitting / max_val
    
    # For non-fitting pairs, use 1e-9
    heuristic = np.where(mask, heuristic_fitting, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic


def _run_aco(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Simulate the ACO process and return the minimum number of bins used by any ant.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Pheromone initialization
    tau = np.ones((n, n))
    
    best_overall = np.inf
    
    rng = np.random.default_rng(seed)
    
    for it in range(n_iters):
        ant_best_bins = np.inf
        ant_best_path = None
        
        for ant in range(n_ants):
            # State
            used = np.zeros(n, dtype=bool)
            bins = []
            current_bin_items = []
            current_bin_sum = 0.0
            
            # Place a random fitting item to start the first bin
            available = np.where(~used)[0]
            if len(available) == 0:
                break
            first_idx = rng.choice(available)
            used[first_idx] = True
            current_bin_items.append(first_idx)
            current_bin_sum = d[first_idx]
            
            while True:
                # Find fitting items
                remaining = capacity - current_bin_sum
                fitting = []
                for j in range(n):
                    if not used[j] and d[j] <= remaining:
                        fitting.append(j)
                
                if len(fitting) == 0:
                    # Open a new bin
                    bins.append(len(current_bin_items))
                    current_bin_items = []
                    current_bin_sum = 0.0
                    
                    # Place a random fitting item
                    available = np.where(~used)[0]
                    if len(available) == 0:
                        break
                    first_idx = rng.choice(available)
                    used[first_idx] = True
                    current_bin_items.append(first_idx)
                    current_bin_sum = d[first_idx]
                    continue
                
                # Pick next item with probability proportional to tau[current, j] * heuristic[current, j]
                # current is the last item placed
                current = current_bin_items[-1]
                weights = np.array([tau[current, j] * heuristic[current, j] for j in fitting])
                total = np.sum(weights)
                if total <= 0:
                    # Fallback to random
                    next_idx = rng.choice(fitting)
                else:
                    probs = weights / total
                    next_idx = rng.choice(fitting, p=probs)
                
                used[next_idx] = True
                current_bin_items.append(next_idx)
                current_bin_sum += d[next_idx]
            
            # Count final bin if it has items
            if len(current_bin_items) > 0:
                bins.append(len(current_bin_items))
            
            n_bins = len(bins)
            if n_bins < ant_best_bins:
                ant_best_bins = n_bins
                ant_best_path = bins
        
        # Update pheromone
        decay = 0.9
        tau *= decay
        # Add pheromone to all entries: 1 / (fewest bins of that iteration)
        if ant_best_bins < np.inf:
            tau += 1.0 / ant_best_bins
        
        if ant_best_bins < best_overall:
            best_overall = ant_best_bins
    
    return best_overall


def _evaluate_heuristic(demands, capacity, heuristic, seed=42):
    """Evaluate a heuristic by running ACO and returning the best bins used."""
    try:
        result = _run_aco(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=seed)
        return result
    except Exception:
        return np.inf


# EVOLVE-BLOCK-END
