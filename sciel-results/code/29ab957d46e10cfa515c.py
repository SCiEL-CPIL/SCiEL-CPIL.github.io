"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after
    item i in the same bin. Larger is more attractive.
    
    Strategy:
    - Pairs that don't fit get 1e-9.
    - Pairs that fit get a score based on how well they complement each other.
    - We want to encourage pairs that together fill the bin well (sum close to capacity)
      but not too close to avoid wasting space that could fit a third item.
    - Actually, in bin packing, we want to pack as much as possible per bin.
    - A good heuristic is to reward pairs whose sum is close to capacity, because
      that means we're using the bin efficiently.
    - However, we also need to consider that after placing two items, we might be
      able to add a third. So we want pairs that leave room for at least one more
      item if possible, but prioritize high utilization.
    
    Let's use a heuristic that rewards the sum being close to capacity:
    - For fitting pairs, compute ratio = (demands[i] + demands[j]) / capacity
    - We want ratio to be as high as possible (close to 1) but not exactly 1
      (since that leaves no room).
    - Actually, in ACO for bin packing, common heuristics reward high utilization.
    - Let's use: heuristic[i,j] = (demands[i] + demands[j]) / capacity for fitting pairs,
      but boost it for very high utilization.
    
    Actually, let's think about this differently. The goal is to minimize the number
    of bins. A good local decision is to place items that together use as much
    capacity as possible. So we want to maximize the sum of the two items, subject
    to the sum being <= capacity.
    
    So a simple heuristic is: for fitting pairs, use (demands[i] + demands[j]) / capacity.
    This rewards pairs that use more of the bin's capacity.
    
    But we can do better. Let's consider that we want to avoid leaving small gaps
    that can't fit any remaining item. However, since we don't know which items
    remain, a simple utilization-based heuristic is reasonable.
    
    Let's use a power function to amplify the preference for high utilization:
    heuristic[i,j] = ((demands[i] + demands[j]) / capacity) ** k for fitting pairs,
    where k > 1 to strongly favor high-utilization pairs.
    
    Let's try k = 2 or 3.
    """
    n = len(demands)
    pair_sum = demands[:, None] + demands[None, :]
    mask = pair_sum <= capacity
    
    # For fitting pairs, use a strong function of utilization
    # Utilization = pair_sum / capacity
    # We want to reward high utilization, so use a power function
    # Let's use (pair_sum / capacity) ** 3 to strongly favor high-utilization pairs
    
    ratio = np.where(mask, pair_sum / capacity, 0)
    heuristic = np.where(mask, ratio ** 3, 1e-9)
    
    return heuristic


def run_aco(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42, time_limit=280):
    """
    Simulates the ACO process and returns the minimum number of bins used by any ant.
    """
    n = len(demands)
    # Pheromone initialization
    tau = np.ones((n, n))
    
    best_overall = np.inf
    start_time = time.time()
    
    rng = np.random.default_rng(seed)
    
    # Pre-compute which items fit together for speed
    # Actually, we need to check dynamically as items are used
    
    for it in range(n_iters):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
            
        ant_results = []
        best_ant_path = None
        best_ant_bins = np.inf
        
        # Run all ants
        for ant in range(n_ants):
            # State
            used = [False] * n
            current_bin = []
            bins = 1
            current_load = 0.0
            path = []  # Store (i, j) pairs for pheromone update
            
            # Place first item randomly among fitting items
            # All items fit in an empty bin, so pick randomly
            first_idx = rng.integers(0, n)
            used[first_idx] = True
            current_bin = [first_idx]
            current_load = demands[first_idx]
            
            # Repeatedly pick next item
            while True:
                # Find all fitting items
                fitting = []
                for j in range(n):
                    if not used[j] and current_load + demands[j] <= capacity:
                        fitting.append(j)
                
                if not fitting:
                    # Open new bin
                    bins += 1
                    # Find an unused item
                    unused = [j for j in range(n) if not used[j]]
                    if not unused:
                        break
                    # Pick randomly from unused
                    next_idx = unused[rng.integers(0, len(unused))]
                    path.append((current_bin[-1], next_idx))
                    used[next_idx] = True
                    current_bin = [next_idx]
                    current_load = demands[next_idx]
                    continue
                
                # Pick with probability proportional to tau[current, j] * heuristic[current, j]
                current = current_bin[-1]
                weights = np.array([tau[current, j] * heuristic[current, j] for j in fitting])
                
                # Ensure weights are positive
                weights = np.where(weights > 0, weights, 1e-10)
                total = weights.sum()
                if total == 0:
                    # Fallback to random
                    next_idx = fitting[rng.integers(0, len(fitting))]
                else:
                    probs = weights / total
                    next_idx = fitting[rng.choice(len(fitting), p=probs)]
                
                path.append((current, next_idx))
                used[next_idx] = True
                current_bin.append(next_idx)
                current_load += demands[next_idx]
            
            ant_results.append(bins)
            if bins < best_ant_bins:
                best_ant_bins = bins
                best_ant_path = path
        
        # Update best overall
        if best_ant_bins < best_overall:
            best_overall = best_ant_bins
        
        # Pheromone update
        tau *= 0.9
        if best_ant_path is not None:
            reward = 1.0 / best_ant_bins
            for (i, j) in best_ant_path:
                tau[i, j] += reward
    
    return best_overall


# The heuristics function is the entry point. The ACO is run by the evaluator
# using the heuristic returned by heuristics().
# We need to make sure the heuristic is computed efficiently.
# The actual ACO simulation is done by the evaluator, not by us.
# We just need to return a good heuristic matrix.
# EVOLVE-BLOCK-END
