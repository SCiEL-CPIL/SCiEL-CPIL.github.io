"""
Strategy: Clarke-Wright Savings Heuristic.
Instead of picking the nearest neighbor, we pick the node that provides the greatest
reduction in total distance compared to serving it separately from the depot.
This is a structural change from local greedy search to a global savings-based construction.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _select_savings_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node based on the Clarke-Wright Savings heuristic.
    We simulate the potential saving of adding each candidate node to the current route.
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    # Find feasible nodes
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if len(feasible) == 0:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    # If time is critical, fall back to fast nearest neighbor
    if _time_remaining() < 0.1:
        d = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(d)])

    # The current node is the "end" of the current route segment.
    # The route started at depot and is currently at current_node.
    # If we add node 'n', the incremental cost change compared to closing the route now
    # and opening a new one for 'n' is:
    # Cost(current path) = ... -> current_node -> depot
    # Cost(new path) = ... -> current_node -> n -> depot (potentially more nodes)
    
    # Standard Savings Heuristic logic:
    # Saving = d(depot, current_node) + d(depot, n) - d(current_node, n)
    # However, since we are extending an existing route, the "saving" is the reduction in total distance
    # by visiting n on this route instead of starting a new route from depot.
    
    # Let's refine the score:
    # If we go to depot now: cost = d(current, depot)
    # If we go to n then depot: cost = d(current, n) + d(n, depot)
    # We want to minimize total length.
    # The "penalty" of extending is (d(current, n) + d(n, depot)) - d(current, depot).
    # Minimizing this penalty is equivalent to maximizing the "saving" where:
    # Saving = d(current, depot) - (d(current, n) + d(n, depot))
    # Wait, standard savings is d(depot, i) + d(depot, j) - d(i, j).
    # Here, i = current_node, j = n.
    # Saving = d(depot, current_node) + d(depot, n) - d(current_node, n).
    
    # We want to pick the node that gives the highest positive saving, or if all are negative,
    # we should probably close the route (return to depot) unless forced by capacity/other constraints.
    # However, the interface forces us to pick a node or depot.
    
    # Let's calculate the incremental cost for each feasible node.
    # Incremental Cost = d(current, n) + d(n, depot) - d(current, depot)
    # We want to minimize this.
    
    d_current_depot = distance_matrix[current_node, depot]
    d_current_feasible = distance_matrix[current_node, feasible]
    d_feasible_depot = distance_matrix[feasible, depot]
    
    incremental_costs = d_current_feasible + d_feasible_depot - d_current_depot
    
    # If the best incremental cost is positive, it means extending the route is more expensive
    # than closing it now and starting fresh.
    # However, we must visit all nodes. Closing now means the *next* route will have to visit these nodes.
    # The savings heuristic globally optimizes this.
    
    # To be safer, let's use a lookahead with the savings heuristic.
    # But first, let's see if a simple savings score works.
    
    # Find the node with the minimum incremental cost (max saving)
    min_cost_idx = np.argmin(incremental_costs)
    best_node = feasible[min_cost_idx]
    min_cost_val = incremental_costs[min_cost_idx]
    
    # If the best extension is still very costly (e.g., > 2 * d_current_depot), 
    # it might be better to close the route?
    # The problem doesn't allow "wait". We must pick next node or depot.
    # If we pick depot, we close the route.
    # If we pick a node, we extend.
    
    # Let's compare the cost of extending vs closing.
    # Cost to close: d(current, depot)
    # Cost to extend to best_node: d(current, best_node) + d(best_node, depot) [optimistic, assuming we go straight to depot after]
    
    # Actually, the "incremental cost" calculated above IS the difference.
    # If min_cost_val > 0, extending is more expensive than closing *immediately* in terms of direct path to depot.
    # But we have to visit the node eventually.
    
    # Let's use a 1-step lookahead to see if extending allows a better subsequent move.
    # This is a hybrid: Savings score for candidates, lookahead for validation.
    
    k = min(5, len(feasible))
    # Sort by incremental cost (savings)
    top_k_indices = np.argsort(incremental_costs)[:k]
    candidates = [feasible[i] for i in top_k_indices]
    
    best_node_final = candidates[0]
    best_total_cost = float('inf')
    
    # Lookahead depth 2
    for cand in candidates:
        sim_current = cand
        sim_remaining = rest_capacity - demands[cand]
        
        # Path so far: current_node -> cand
        path_cost = distance_matrix[current_node, cand]
        
        sim_unvisited = [n for n in unvisited_nodes if n != cand]
        
        # Try to add one more node
        if sim_unvisited and sim_remaining > 0:
            sim_feasible = [n for n in sim_unvisited if demands[n] <= sim_remaining]
            if sim_feasible:
                # Use savings heuristic for the next step too
                d_sim_depot = distance_matrix[sim_current, depot]
                d_sim_f = distance_matrix[sim_current, sim_feasible]
                d_f_depot = distance_matrix[sim_feasible, depot]
                inc_costs = d_sim_f + d_f_depot - d_sim_depot
                
                next_best_idx = np.argmin(inc_costs)
                next_node = sim_feasible[next_best_idx]
                
                # Total cost for this branch: current->cand->next->depot
                total_cost = path_cost + distance_matrix[sim_current, next_node] + distance_matrix[next_node, depot]
                
                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    best_node_final = cand
            else:
                # No feasible next node, so route ends at cand
                total_cost = path_cost + distance_matrix[sim_current, depot]
                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    best_node_final = cand
        else:
            # No capacity or no nodes left, route ends at cand
            total_cost = path_cost + distance_matrix[sim_current, depot]
            if total_cost < best_total_cost:
                best_total_cost = total_cost
                best_node_final = cand

    return int(best_node_final)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Main entry point.
    """
    if _time_remaining() < 0.1:
        # Fallback to fast nearest feasible
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        d = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(d)])

    if len(unvisited_nodes) == 0:
        return depot

    return _select_savings_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)

# EVOLVE-BLOCK-END
