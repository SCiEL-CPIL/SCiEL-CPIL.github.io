"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache
_dm_cache = {}
_start_time = None
_total_unvisited_initial = 50 # Assumed based on task description

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix, dtype=np.float64)
    return _dm_cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    # Safety margin of 5 seconds
    return (time.time() - _start_time) > 145

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: Adaptive Savings-Weighted Nearest Neighbor.
    - Early in the route: Prioritize high savings nodes to form compact clusters.
    - Late in the route: Prioritize nodes close to the depot to minimize return cost.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Deadline guard: fall back to nearest feasible
    if _check_deadline():
        feasible = [n for n in unvisited if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        feasible_arr = np.array(feasible)
        dists = dm[current_node, feasible_arr]
        best_idx = np.argmin(dists)
        return int(feasible_arr[best_idx])
    
    # Filter to capacity-feasible nodes
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    feasible_arr = np.array(feasible, dtype=np.int64)
    n = len(feasible_arr)
    
    # Compute distances
    dists_to_current = dm[current_node, feasible_arr]
    dists_to_depot = dm[feasible_arr, depot]
    
    # Sub-matrix of distances among feasible nodes
    sub_dm = dm[np.ix_(feasible_arr, feasible_arr)]
    
    # Calculate Savings for each node i
    # Savings(i) = sum_{j != i} (d(depot, i) + d(depot, j) - d(i, j))
    # = (n-1)*d(depot, i) + sum_{j != i} d(depot, j) - sum_{j != i} d(i, j)
    
    # sum_{j != i} d(i, j) is the row sum of sub_dm (excluding diagonal)
    row_sums = np.sum(sub_dm, axis=1)
    
    # sum_{j != i} d(depot, j) is constant for all i, so it doesn't affect ranking
    # Let's just use: Savings(i) proportional to -row_sums[i] + (n-1)*dists_to_depot[i]
    # Higher savings is better.
    # We want to MINIMIZE score.
    # So we subtract savings from the score.
    
    # Normalize components
    max_dtc = np.max(dists_to_current) if np.max(dists_to_current) > 0 else 1.0
    norm_dtc = dists_to_current / max_dtc
    
    max_ddep = np.max(dists_to_depot) if np.max(dists_to_depot) > 0 else 1.0
    norm_ddep = dists_to_depot / max_ddep
    
    # Normalize savings potential
    # savings_potential = (n-1)*dists_to_depot - row_sums
    savings_potential = (n - 1) * dists_to_depot - row_sums
    max_sp = np.max(savings_potential) if np.max(savings_potential) > 0 else 1.0
    norm_sp = savings_potential / max_sp
    
    # Adaptive Weights
    # Estimate progress: how many nodes are left vs initial
    # We don't know the exact initial count per instance if it varies, but task says 50.
    # Even if it varies, the ratio len(feasible)/50 is a good proxy for "remaining fraction".
    # However, capacity constraints might reduce feasible set size.
    # Let's use the fraction of feasible nodes remaining relative to a typical route size.
    # A safer metric: distance from current to depot.
    # If d(current, depot) is small, we are near start.
    # If d(current, depot) is large, we are near end.
    
    d_current_depot = dm[current_node, depot]
    
    # Normalize d_current_depot against max possible (diagonal of dm? or max dist in matrix?)
    # Use max dist in the feasible set to depot as a scale?
    # Or just use a fixed scale if distances are normalized (e.g., 0-1 or 0-100).
    # Let's assume distances are roughly in [0, 100] or similar.
    # A robust scale: max(dists_to_depot)
    max_ddep_global = np.max(dists_to_depot) if np.max(dists_to_depot) > 0 else 1.0
    
    # If current is close to depot, we are "early".
    # If current is far from depot, we are "late".
    # Let's define a phase factor p in [0, 1].
    # p = 0 at start, p = 1 at end.
    # Approximation: p = d_current_depot / max_ddep_global
    # This is not perfect but a reasonable heuristic.
    
    p = d_current_depot / max_ddep_global if max_ddep_global > 0 else 0.0
    p = np.clip(p, 0.0, 1.0)
    
    # Weights
    # Early (p=0): High weight on Savings (norm_sp), Low weight on Depot Dist (norm_ddep).
    # Late (p=1): Low weight on Savings, High weight on Depot Dist.
    
    # Base weights
    w_dtc = 1.0
    w_ddep = 0.5 + 0.5 * p  # 0.5 to 1.0
    w_sp = 0.8 * (1.0 - p)  # 0.8 to 0.0
    
    # Score = w_dtc * norm_dtc + w_ddep * norm_ddep - w_sp * norm_sp
    scores = w_dtc * norm_dtc + w_ddep * norm_ddep - w_sp * norm_sp
    
    # Pick the feasible node with the lowest score
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
