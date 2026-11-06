"""Rewritten program: Savings-based (Clarke-Wright) heuristic with soft-min lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Time guard: track when the process started
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill


def _time_exceeded():
    return (time.time() - _START_TIME) > _TIME_LIMIT


def _soft_min(values, temp):
    """Smooth minimum via log-sum-exp trick. Returns a value close to min(values)."""
    if len(values) == 0:
        return 0.0
    v = np.asarray(values, dtype=float)
    v_min = np.min(v)
    # If all values are equal or very close, just return the min
    if np.allclose(v, v_min):
        return float(v_min)
    # Use shifted values for numerical stability
    shifted = v - v_min
    # soft_min = -temp * log(sum(exp(-shifted/temp))) + v_min
    # = v_min - temp * log(sum(exp(-shifted/temp)))
    # For numerical stability with large values:
    neg_shifted = -shifted / temp
    max_neg = np.max(neg_shifted)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_shifted - max_neg)))
    return float(v_min - temp * log_sum)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer (or depot 0 to close) using a savings-based heuristic
    with soft-min lookahead for next-step cost estimation.
    
    Score components:
    1. Savings (Clarke-Wright): dist(depot,cur) + dist(depot,j) - dist(cur,j)
    2. Soft-min lookahead: approximate cost of the best next step after visiting j
    3. Capacity utilization bonus: prefer nodes that fill the route more
    """
    if _time_exceeded():
        # Fallback: just return nearest feasible to be safe
        unvisited = np.asarray(unvisited_nodes, dtype=int)
        if len(unvisited) == 0:
            return 0
        feasible = unvisited[demands[unvisited] <= rest_capacity]
        if len(feasible) == 0:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    unvisited = np.asarray(unvisited_nodes, dtype=int)
    n_unvisited = len(unvisited)

    if n_unvisited == 0:
        return 0

    # Feasibility: demand must fit in remaining capacity
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible = unvisited[feasible_mask]

    if len(feasible) == 0:
        return 0

    # If only one feasible node, take it
    if len(feasible) == 1:
        return int(feasible[0])

    # Precompute distance rows for efficiency
    dist_depot = distance_matrix[depot]
    dist_cur = distance_matrix[current_node]
    
    dist_depot_current = dist_depot[current_node]
    dist_depot_feasible = dist_depot[feasible]
    dist_current_feasible = dist_cur[feasible]

    # Component 1: Savings (higher is better)
    # savings[j] = dist(depot, cur) + dist(depot, j) - dist(cur, j)
    savings = dist_depot_current + dist_depot_feasible - dist_current_feasible

    # Component 2: Soft-min lookahead
    # After visiting j, the best next step cost is approximately:
    #   min over k in feasible\{j} of dist(j,k) + dist(k,depot) [if we go to depot after]
    #   or min over k in feasible\{j} of dist(j,k) [if we continue]
    # We approximate the marginal cost of the next step after j.
    # Use a soft-min over distances from j to other feasible nodes + return to depot.
    # This is O(n^2) but n is at most 50, so it's fine.
    
    n_feasible = len(feasible)
    if n_feasible > 1:
        # For each feasible j, compute soft-min of [dist(j,k) + dist(k,depot) for k != j]
        # This approximates the cost of the next leg after j
        # dist_matrix[feasible, feasible] gives pairwise distances
        D_ff = distance_matrix[np.ix_(feasible, feasible)]
        # dist from each feasible to depot
        d_to_depot = dist_depot_feasible  # shape (n_feasible,)
        
        # For each j (row i), the cost of going to k (col j) then depot is D_ff[i,j] + d_to_depot[j]
        # We want soft-min over k != i of (D_ff[i,k] + d_to_depot[k])
        # Add depot return cost
        next_costs = D_ff + d_to_depot[np.newaxis, :]  # shape (n_feasible, n_feasible)
        
        # Mask out diagonal (can't go to self)
        np.fill_diagonal(next_costs, np.inf)
        
        # Soft-min along axis=1 for each row
        temp = 2.0
        lookahead_costs = np.zeros(n_feasible)
        for i in range(n_feasible):
            row = next_costs[i]
            finite_mask = np.isfinite(row)
            if np.any(finite_mask):
                lookahead_costs[i] = _soft_min(row[finite_mask], temp)
            else:
                # Only option is to go back to depot
                lookahead_costs[i] = dist_current_feasible[i]
    else:
        # Only one feasible node, already handled above
        lookahead_costs = np.zeros(len(feasible))

    # We want to MINIMIZE lookahead cost, so negate it for scoring
    # Scale to be comparable with savings
    max_savings = np.max(savings) if np.max(savings) > 0 else 1.0
    max_lookahead = np.max(lookahead_costs) if np.max(lookahead_costs) > 0 else 1.0
    
    # Component 3: Capacity utilization bonus
    # Prefer nodes that use more of the remaining capacity (reduces number of routes)
    # Bonus proportional to demand[j] / rest_capacity
    cap_bonus = demands[feasible].astype(float) / max(rest_capacity, 1)
    
    # Composite score: maximize
    # savings_weight * normalized_savings - lookahead_weight * normalized_lookahead + cap_weight * cap_bonus
    # Normalize savings to [0, 1]
    min_sav = np.min(savings)
    sav_range = max_savings - min_sav if max_savings > min_sav else 1.0
    norm_savings = (savings - min_sav) / sav_range
    
    # Normalize lookahead to [0, 1] (lower is better, so we negate)
    min_la = np.min(lookahead_costs)
    la_range = max_lookahead - min_la if max_lookahead > min_la else 1.0
    norm_lookahead = (lookahead_costs - min_la) / la_range
    
    # Weights
    w_savings = 1.0
    w_lookahead = 0.3
    w_cap = 0.1
    
    score = w_savings * norm_savings - w_lookahead * norm_lookahead + w_cap * cap_bonus

    best_idx = int(np.argmax(score))
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
