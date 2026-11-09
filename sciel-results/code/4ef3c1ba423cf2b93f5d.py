"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache
_dm_cache = {}
_start_time = None
_dm_array = None

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    global _dm_array
    if _dm_array is None:
        _dm_array = np.asarray(distance_matrix, dtype=np.float64)
    return _dm_array

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
    
    Strategy: 2-step lookahead with savings-based scoring.
    For each feasible candidate, estimate the cost of visiting it now plus
    the best subsequent move, combined with a savings potential term.
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
    
    # Compute distances from current node to all feasible nodes
    dists_to_current = dm[current_node, feasible_arr]
    
    # Compute distances from each feasible node to depot
    dists_to_depot = dm[feasible_arr, depot]
    
    # Sub-matrix of distances among feasible nodes
    sub_dm = dm[np.ix_(feasible_arr, feasible_arr)]
    
    if n > 1:
        # For 2-step lookahead: for each candidate i, find the best next node j
        # (j != i, j feasible with remaining capacity after i)
        # Approximate: for each i, the best next is min over j of d(i,j) + d(j,depot)
        # This approximates: go to i, then to j, then to depot (or continue)
        
        # For each i, compute min over j != i of (d(i,j) + d(j,depot))
        # This is a simple 2-step estimate
        # We need to be careful: after visiting i, remaining capacity is rest_capacity - demands[i]
        # For simplicity in the lookahead, we assume all remaining feasible nodes are still reachable
        # (which is an approximation, but fast)
        
        # Cost if we go to i then to best j then to depot:
        # d(current,i) + min_j (d(i,j) + d(j,depot))
        # But we also need to consider that after j, there are still other nodes.
        # For a 2-step heuristic, this is a reasonable approximation.
        
        # Compute for each i: min over j != i of (sub_dm[i,j] + dists_to_depot[j])
        # To exclude self, set diagonal to inf temporarily
        sub_dm_copy = sub_dm.copy()
        np.fill_diagonal(sub_dm_copy, np.inf)
        
        # For each row i, add dists_to_depot to each column j
        # Then take min over j
        two_step_costs = sub_dm_copy + dists_to_depot[None, :]
        best_two_step = np.min(two_step_costs, axis=1)
        
        # Total 2-step cost for candidate i: d(current,i) + best_two_step[i]
        two_step_total = dists_to_current + best_two_step
        
        # Savings-based score: prefer nodes that have high savings with others
        # savings(i,j) = d(depot,i) + d(depot,j) - d(i,j)
        d_depot = dists_to_depot
        savings_matrix = d_depot[:, None] + d_depot[None, :] - sub_dm
        
        # Total savings for each node with all others
        total_savings = np.sum(savings_matrix, axis=1)
        
        # Normalize savings
        max_savings = np.max(total_savings) if np.max(total_savings) > 0 else 1.0
        norm_savings = total_savings / max_savings
        
        # Normalize 2-step cost
        max_two_step = np.max(two_step_total) if np.max(two_step_total) > 0 else 1.0
        norm_two_step = two_step_total / max_two_step
        
        # Composite score: minimize
        # - norm_two_step: immediate + next step cost (weight: 1.0)
        # - alpha * norm_savings: prefer high savings (weight: 0.4)
        alpha = 0.4
        scores = norm_two_step - alpha * norm_savings
        
    else:
        return int(feasible_arr[0])
    
    # Pick the feasible node with the lowest score
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
