"""Capacitated vehicle routing: pick the next customer or return to depot."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Cache numpy views of the distance matrix to avoid repeated conversion cost.
_DM_CACHE = {}
_DEMAND_CACHE = {}


def _get_dm(distance_matrix):
    key = id(distance_matrix)
    dm = _DM_CACHE.get(key)
    if dm is None:
        dm = np.asarray(distance_matrix, dtype=np.float64)
        _DM_CACHE[key] = dm
    return dm


def _get_demands(demands):
    key = id(demands)
    d = _DEMAND_CACHE.get(key)
    if d is None:
        d = np.asarray(demands, dtype=np.float64)
        _DEMAND_CACHE[key] = d
    return d


# Deadline guard: the evaluator kills the process at 150s wall-clock.
_T0 = time.perf_counter()
_DEADLINE = _T0 + 140.0  # 10s safety margin


def _deadline_exceeded():
    return time.perf_counter() > _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Safety: if we are about to be killed, just close the route quickly.
    if _deadline_exceeded():
        return int(depot)

    dm = _get_dm(distance_matrix)
    dem = _get_demands(demands)

    unvisited = np.asarray(unvisited_nodes, dtype=np.int64)
    n = unvisited.size

    if n == 0:
        return int(depot)

    # Feasibility: candidate demand must fit in remaining capacity.
    cand_dem = dem[unvisited]
    feasible_mask = cand_dem <= rest_capacity

    # If nothing fits, we must return to depot.
    if not np.any(feasible_mask):
        return int(depot)

    if np.count_nonzero(feasible_mask) == n:
        cand = unvisited
        cand_dem = cand_dem
    else:
        cand = unvisited[feasible_mask]
        cand_dem = cand_dem[feasible_mask]

    k = len(cand)
    if k == 1:
        return int(cand[0])

    d_cur_cand = dm[current_node, cand]
    d_cand_depot = dm[cand, depot]

    # --- Savings-based / Clarke-Wright inspired heuristic ---
    # The "savings" of visiting candidate c next (rather than returning to depot now
    # and later visiting c from depot) is:
    #   saving(c) = d(cur, depot) + d(depot, c) - d(cur, c)
    # Higher savings mean more beneficial to visit c now.
    # We also incorporate a 1-step lookahead: after visiting c, what's the best
    # next move? This captures local structure.
    
    d_cur_depot = dm[current_node, depot]
    
    # Base savings
    savings = d_cur_depot + d_cand_depot - d_cur_cand

    # 1-step lookahead: for each candidate c, estimate the best next cost
    # after visiting c. The next step will be either another feasible customer
    # or the depot. We approximate by the min distance from c to any other
    # feasible candidate or to depot.
    cand_dm = dm[np.ix_(cand, cand)]
    np.fill_diagonal(cand_dm, np.inf)
    
    # For each candidate, the minimum distance to any other candidate
    min_dist_to_other = np.min(cand_dm, axis=1)
    
    # After visiting c, the "next leg" cost is min(dist to other feasible, dist to depot)
    # But we also need to consider capacity: after visiting c, remaining capacity
    # decreases by cand_dem[c]. So not all other candidates may still be feasible.
    # Approximation: use min of all other candidates (slightly optimistic)
    next_leg_cost = np.minimum(min_dist_to_other, d_cand_depot)
    
    # Total estimated route cost if we pick c now:
    #   d(cur, c) + next_leg_cost + d(next, depot)  [roughly]
    # But we want to MINIMIZE total route length, so we want to minimize this.
    # Alternatively, maximize savings and minimize next_leg_cost.
    
    # Combined score: we want to minimize
    #   score = d_cur_cand + next_leg_cost - alpha * savings
    # where alpha controls how much we value the savings term.
    
    # Normalize savings to be on similar scale as distances
    # Use a weight that balances the terms
    alpha = 0.3
    
    score = d_cur_cand + next_leg_cost - alpha * savings
    
    # Add a small demand-based tiebreaker: prefer higher demand to fill vehicle
    # This helps reduce the number of vehicles needed
    score = score - 0.05 * cand_dem
    
    best_idx = int(np.argmin(score))
    return int(cand[best_idx])
# EVOLVE-BLOCK-END
