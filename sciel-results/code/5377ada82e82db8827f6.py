"""Direct heuristic for TSP next-node selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # safety margin before 150s hard kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction.
    
    Direct approach:
    - Use a blend of distance from current node and distance to the destination (start node).
    - The weight on destination increases as the tour nears completion to avoid a long closing edge.
    - For small remaining sets, use a 2-step lookahead that considers the cost of going to a candidate,
      then to the best next node, then to destination.
    
    This is a direct, formula-driven heuristic derived from the task structure:
    we want to minimize total tour length, so at each step we balance local progress
    (going to a nearby node) with global completion (ending near the destination).
    """
    # Time guard
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        # For two nodes left, pick the one that minimizes total edge cost to close the tour
        u, v = unvisited_nodes
        # Option 1: go to u then v: cost = d(current,u) + d(u,v) + d(v,dest)
        cost_u = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
        # Option 2: go to v then u: cost = d(current,v) + d(v,u) + d(u,dest)
        cost_v = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
        return int(unvisited_nodes[0] if cost_u <= cost_v else unvisited_nodes[1])
    
    # Adaptive weight: more emphasis on destination as we get closer to finishing
    # Strategy: Pure NN for the first ~40 nodes, then blend in destination proximity.
    # This avoids distorting the early tour construction while ensuring a short closing edge.
    
    if n_unvisited > 40:
        alpha = 0.0
    else:
        # Smoothly increase alpha from 0 to 0.8 as n_unvisited goes from 40 to 1
        # Using a quadratic ramp to keep it very low until the very end
        progress = (40.0 - n_unvisited) / 39.0
        progress = max(0.0, min(1.0, progress))
        alpha = 0.8 * (progress ** 2)
    
    # Distances from current to all unvisited
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # Distances from each unvisited to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # For small remaining sets, use a 2-step lookahead
    if n_unvisited <= 15:
        # For each candidate i, compute: d(current, i) + min_{j != i} (d(i, j) + d(j, dest))
        # This is a 2-step lookahead that considers the immediate next step and the closing edge
        lookahead_costs = np.zeros(n_unvisited)
        for i in range(n_unvisited):
            c = unvisited_nodes[i]
            cost_to_c = distance_matrix[current_node, c]
            # Find the best next node j (j != i) to minimize d(c, j) + d(j, dest)
            if n_unvisited > 1:
                # Get distances from c to all unvisited except c
                d_from_c = distance_matrix[c, unvisited_nodes]
                d_from_c[i] = np.inf  # exclude self
                # Add destination cost for each potential next node
                future_costs = d_from_c + d_dest
                best_future = np.min(future_costs)
            else:
                best_future = 0.0
            lookahead_costs[i] = cost_to_c + best_future
        
        # Use the lookahead costs directly as the primary score
        scores = lookahead_costs
    else:
        # For larger sets, use the weighted blend
        scores = d_curr + alpha * d_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
