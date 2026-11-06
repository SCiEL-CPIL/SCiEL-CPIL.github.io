"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_DEADLINE = 115.0  # 5s margin before 120s

def _init_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated VRP: pick next customer or depot (0) to close route.
    
    Strategy:
      1. Filter unvisited nodes that fit in rest_capacity.
      2. If none fit, return depot.
      3. Score each feasible node using a combined metric:
         - radial expansion penalty (geometric containment)
         - immediate distance from current_node
         - savings penalty (extra cost vs going straight to depot)
         - 1-step soft-min lookahead
         - capacity utilization bonus
    """
    _init_deadline()
    if _time_exceeded():
        return depot

    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot

    # Filter by capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return depot

    feasible_arr = np.array(feasible, dtype=np.int64)
    n_feasible = len(feasible_arr)
    if n_feasible == 1:
        return int(feasible_arr[0])

    # Distance matrix
    dm = distance_matrix

    # Immediate distances from current_node
    dist_from_current = dm[current_node, feasible_arr]

    # Distances to depot
    dist_to_depot = dm[feasible_arr, depot]
    dist_current_to_depot = dm[current_node, depot]

    # --- Radial expansion penalty (Orbit-Expansion) ---
    # Penalize nodes that significantly expand the bounding circle radius.
    # The "radius" of the current route context is approximated by the distance
    # from depot to current_node. A candidate that is much farther from depot
    # than the current context expands the route's radial footprint.
    # We compute: max(0, dist_to_depot - dist_current_to_depot) as the expansion.
    # This encourages staying within the current "orbit" of the route.
    radial_expansion = np.maximum(0.0, dist_to_depot - dist_current_to_depot)
    
    # Also consider: if we're near the depot (start of route), we should be
    # more tolerant of expansion since we're building the route.
    # Weight the radial penalty by how far we are from depot (proxy for route maturity).
    route_maturity = np.clip(dist_current_to_depot / 100.0, 0.0, 1.0)
    radial_penalty = route_maturity * radial_expansion

    # --- Savings penalty ---
    # Classic savings: d(current, j) + d(j, depot) - d(current, depot)
    savings_penalty = dist_from_current + dist_to_depot - dist_current_to_depot

    # --- 1-step soft-min lookahead ---
    # For each feasible node, compute a smooth surrogate for the minimum distance
    # to any other feasible node or depot (excluding self).
    indices = np.concatenate([feasible_arr, [depot]])
    sub_dist = dm[np.ix_(indices, indices)]

    t = 1.5  # temperature for soft-min (lower = sharper)
    diag_mask = np.eye(n_feasible + 1, dtype=bool)
    masked_sub = sub_dist.copy()
    masked_sub[diag_mask] = 1e9

    # Numerical stability
    row_min = np.min(masked_sub, axis=1)
    shifted = masked_sub - row_min[:, np.newaxis]
    exp_vals = np.exp(-shifted / t)
    exp_vals[diag_mask] = 0.0
    s = np.sum(exp_vals, axis=1)
    lookahead_1 = row_min[:n_feasible] + t * np.log(s[:n_feasible])

    # --- Capacity utilization bonus ---
    # Prefer nodes that use more of the remaining capacity
    cap_bonus = -0.12 * (demands[feasible_arr] / rest_capacity)

    # --- Combined score ---
    # Weights tuned to balance geometric containment, proximity, savings,
    # lookahead, and capacity filling.
    w_radial = 0.35
    w_proximity = 0.20
    w_savings = 0.20
    w_lookahead = 0.15
    w_cap = 1.0  # cap_bonus already includes its weight

    scores = (w_radial * radial_penalty +
              w_proximity * dist_from_current +
              w_savings * savings_penalty +
              w_lookahead * lookahead_1 +
              cap_bonus)

    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
