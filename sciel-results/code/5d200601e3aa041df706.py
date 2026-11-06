"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_DEADLINE = None
_DEADLINE_LIMIT = 115.0  # Safety margin before 120s limit


def _init_deadline():
    global _DEADLINE
    _DEADLINE = time.time() + _DEADLINE_LIMIT


def _time_exceeded():
    global _DEADLINE
    if _DEADLINE is None:
        _init_deadline()
    return time.time() >= _DEADLINE


def _safety_return(unvisited_nodes, rest_capacity, demands, current_node, depot, distance_matrix):
    """Fallback: return nearest feasible node or depot."""
    if len(unvisited_nodes) == 0:
        return depot
    unvisited = np.array(unvisited_nodes)
    feasible_mask = demands[unvisited] <= rest_capacity
    if not np.any(feasible_mask):
        return depot
    feasible_nodes = unvisited[feasible_mask]
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    dists = distance_matrix[current_node, feasible_nodes]
    return int(feasible_nodes[np.argmin(dists)])


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Initialize deadline on first call
    global _DEADLINE
    if _DEADLINE is None:
        _init_deadline()
    
    # Check time
    if _time_exceeded():
        return _safety_return(unvisited_nodes, rest_capacity, demands, current_node, depot, distance_matrix)
    
    if len(unvisited_nodes) == 0:
        return depot
    
    unvisited = np.array(unvisited_nodes)
    
    # Filter to feasible nodes (demand fits in remaining capacity)
    feasible_mask = demands[unvisited] <= rest_capacity
    if not np.any(feasible_mask):
        return depot
    
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # If time is getting tight, use simple heuristic
    if _time_exceeded():
        return _safety_return(unvisited_nodes, rest_capacity, demands, current_node, depot, distance_matrix)
    
    n = len(feasible_nodes)
    demands_arr = demands[feasible_nodes]
    dist_to_current = distance_matrix[current_node, feasible_nodes]
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Multi-factor scoring with 2-step lookahead
    scores = np.zeros(n)
    
    for i in range(n):
        # Immediate cost: distance from current to candidate i
        cost_i = dist_to_current[i]
        
        # Remaining capacity after visiting i
        rest_cap_after = rest_capacity - demands_arr[i]
        
        # 2-step lookahead: estimate cost of next move after i
        if rest_cap_after <= 0:
            # Must return to depot
            lookahead_1 = dist_to_depot[i]
            lookahead_2 = 0.0
        else:
            # Find other feasible nodes with reduced capacity
            other_mask = np.ones(n, dtype=bool)
            other_mask[i] = False
            other_feasible = feasible_nodes[other_mask]
            other_demands = demands_arr[other_mask]
            
            if len(other_feasible) == 0:
                lookahead_1 = dist_to_depot[i]
                lookahead_2 = 0.0
            else:
                feasible_after_mask = other_demands <= rest_cap_after
                if not np.any(feasible_after_mask):
                    lookahead_1 = dist_to_depot[i]
                    lookahead_2 = 0.0
                else:
                    feasible_after = other_feasible[feasible_after_mask]
                    # Distance from candidate i to nearest feasible-after node
                    dists_from_i = distance_matrix[feasible_nodes[i], feasible_after]
                    min_next_dist = np.min(dists_from_i)
                    
                    # Choose between going to next customer or returning to depot
                    # Weighted combination to balance exploration vs. return
                    w_next = 0.6
                    w_depot = 0.4
                    lookahead_1 = w_next * min_next_dist + w_depot * dist_to_depot[i]
                    
                    # 2nd step: if we go to next customer, estimate return cost
                    if min_next_dist < dist_to_depot[i]:
                        # We're likely going to next customer, so estimate return from there
                        next_node = feasible_after[np.argmin(dists_from_i)]
                        rest_cap_2 = rest_cap_after - demands[next_node]
                        if rest_cap_2 <= 0:
                            lookahead_2 = distance_matrix[next_node, depot]
                        else:
                            # Estimate remaining route cost from next_node
                            remaining_feasible_mask = demands[unvisited] <= rest_cap_2
                            remaining_feasible = unvisited[remaining_feasible_mask]
                            if len(remaining_feasible) > 0:
                                # Average distance from next_node to remaining feasible nodes
                                avg_dist = np.mean(distance_matrix[next_node, remaining_feasible])
                                # Rough estimate: proportional to number of remaining nodes
                                lookahead_2 = 0.3 * avg_dist + 0.7 * distance_matrix[next_node, depot]
                            else:
                                lookahead_2 = distance_matrix[next_node, depot]
                    else:
                        # We're likely returning to depot, so no 2nd step
                        lookahead_2 = 0.0
        
        # Combine immediate cost and lookahead
        scores[i] = cost_i + 0.7 * lookahead_1 + 0.3 * lookahead_2
    
    # Add capacity utilization bonus: prefer nodes that use more capacity when close in score
    # This helps fill vehicles efficiently
    cap_util = demands_arr / max(rest_capacity, 1)
    scores -= 0.05 * cap_util
    
    # Add small penalty for very high demand nodes to avoid getting stuck
    high_demand_penalty = np.where(demands_arr > 6, 0.02 * (demands_arr - 6), 0)
    scores += high_demand_penalty
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
