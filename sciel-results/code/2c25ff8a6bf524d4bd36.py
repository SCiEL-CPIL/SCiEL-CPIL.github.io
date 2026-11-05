"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 145.0


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard
    if not _time_ok():
        if len(unvisited_nodes) == 0:
            return int(depot)
        # Emergency: nearest feasible
        unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
        feas_mask = demands[unvisited_arr] <= rest_capacity
        if not np.any(feas_mask):
            return int(depot)
        feas = unvisited_arr[feas_mask]
        return int(feas[np.argmin(distance_matrix[current_node, feas])])

    if len(unvisited_nodes) == 0:
        return int(depot)

    # Convert to numpy array
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    
    # Filter feasible nodes based on capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)
    
    feas_nodes = unvisited_arr[feas_mask]
    
    if len(feas_nodes) == 1:
        return int(feas_nodes[0])
    
    # Distances
    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_feas = distance_matrix[depot, feas_nodes]
    d_cur_feas = distance_matrix[current_node, feas_nodes]
    
    # Savings heuristic:
    # Saving = d[current, depot] + d[depot, c] - d[current, c]
    # We want to maximize this.
    savings = d_cur_dep + d_dep_feas - d_cur_feas
    
    # Adaptive penalty: when capacity is tight, we should prefer nodes that
    # allow us to pack more customers. A high-demand node when rest_capacity is
    # small means we're likely closing the route soon, so we should prefer
    # nodes that are close to the depot (to minimize closing cost).
    # 
    # The ratio rest_capacity / max_demand_in_feasible indicates how "tight" we are.
    # If tight, add a penalty proportional to distance from depot.
    max_demand = np.max(demands[feas_nodes])
    if max_demand > 0:
        tightness = 1.0 - (rest_capacity / max_demand)
        # Clamp tightness to [0, 1]
        tightness = np.clip(tightness, 0.0, 1.0)
    else:
        tightness = 0.0
    
    # Base score: negative savings (we want to minimize, so negate)
    # Plus adaptive penalty for being far from depot when tight
    # penalty = tightness * d_dep_feas
    # score = -savings + penalty
    
    # But we also want to ensure we don't pick a node that is very far from current
    # even if savings are high. Let's combine:
    # score = -savings + 0.3 * d_cur_feas + tightness * 0.5 * d_dep_feas
    
    # Actually, let's stick closer to the pure savings but add a small
    # distance-to-current penalty to avoid zig-zagging, and the tightness penalty.
    
    # Primary: maximize savings (minimize -savings)
    # Secondary: minimize distance to current (to keep route compact)
    # Tertiary: minimize distance to depot when capacity is tight
    
    # score = -savings + 0.1 * d_cur_feas + tightness * 0.3 * d_dep_feas
    
    scores = -savings + 0.1 * d_cur_feas + tightness * 0.3 * d_dep_feas
    
    best_idx = np.argmin(scores)
    best_node = feas_nodes[best_idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
