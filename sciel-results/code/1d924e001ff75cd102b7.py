"""TSP constructive heuristic: regret-based selection with adaptive lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import permutations

_start_time = time.time()
_LIMIT = 110.0  # Safety margin before 120s limit


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a regret-based heuristic with adaptive lookahead.
    
    The regret heuristic selects the node whose second-best alternative is most
    costly compared to its best alternative. This encourages early commitment
    to nodes that have few good alternatives, preventing them from becoming
    isolated later in the tour.
    
    For small sets (<=6), use exact brute force to optimize the tail.
    """
    n_unvisited = len(unvisited_nodes)
    
    # Deadline check
    if _is_deadline_passed():
        # Return nearest node as fallback
        if n_unvisited == 0:
            return destination_node
        dists = distance_matrix[current_node, unvisited_nodes]
        idx = np.argmin(dists)
        return int(unvisited_nodes[idx])
    
    # If no unvisited nodes, return destination
    if n_unvisited == 0:
        return destination_node
    
    # If only 1 node left, return it
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If few nodes left, use exact brute force to optimize the tail
    if n_unvisited <= 6:
        nodes = list(unvisited_nodes)
        best_cost = np.inf
        best_first_node = nodes[0]
        
        for perm in permutations(nodes):
            # Cost: current -> perm[0] -> ... -> perm[-1] -> destination
            cost = distance_matrix[current_node, perm[0]]
            for i in range(len(perm) - 1):
                cost += distance_matrix[perm[i], perm[i+1]]
            cost += distance_matrix[perm[-1], destination_node]
            
            if cost < best_cost:
                best_cost = cost
                best_first_node = perm[0]
                
        return int(best_first_node)
    
    # For larger sets, use regret-based heuristic
    # Compute distance from current node to all unvisited
    dist_curr = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distance from destination to all unvisited
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    
    # Compute pairwise distances among unvisited nodes
    D = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each unvisited node i, compute the cost of inserting it:
    # cost_i = dist_curr[i] + min_j(dist_matrix[i,j]) where j != i, j in unvisited
    # This approximates the marginal cost of choosing i now
    
    # Exclude self by setting diagonal to inf
    D_no_self = D.copy()
    np.fill_diagonal(D_no_self, np.inf)
    
    # Min distance from each unvisited node to any other unvisited node
    min_to_other = np.min(D_no_self, axis=1)
    
    # Immediate cost of choosing node i
    immediate_cost = dist_curr + min_to_other
    
    # Regret: difference between second-best and best immediate cost
    # Sort immediate costs for each node
    sorted_costs = np.sort(immediate_cost)
    best_cost = sorted_costs[:, 0]
    second_best_cost = sorted_costs[:, 1]
    
    # Regret = second_best - best (higher regret means more critical to decide now)
    regret = second_best_cost - best_cost
    
    # Also consider destination awareness: penalize nodes far from destination
    # to help close the tour
    dest_penalty = dest_dist * 0.3
    
    # Combine regret and destination penalty
    # Higher regret -> more urgent; lower dest_penalty -> better
    score = -regret + dest_penalty
    
    # Select node with minimum score (highest regret, lowest dest penalty)
    idx = np.argmin(score)
    
    return int(unvisited_nodes[idx])
# EVOLVE-BLOCK-END
