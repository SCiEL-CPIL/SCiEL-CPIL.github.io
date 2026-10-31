"""
TSP constructive heuristic: Regret-2 with Destination Awareness.

This program implements a constructive heuristic for the Traveling Salesman Problem.
Instead of using expensive MST calculations or minimax penalties, it uses a 
Regret-2 strategy that looks two steps ahead:
1. Immediate cost to reach candidate u.
2. Cost to go from u to the best next node v, and then to the destination.

This balances local greediness with global connectivity to the destination,
avoiding the high computational cost of MST updates while maintaining
strong performance on Euclidean instances.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline enforcement
_start_time = time.time()
_TOTAL_LIMIT = 150.0  # Evaluator limit
_MARGIN = 10.0        # Safety margin

def _is_deadline_passed():
    """Check if we are close to the time limit."""
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour using Regret-2 heuristic.
    
    Strategy:
    For each candidate u in unvisited_nodes:
    1. c1(u) = d(current, u)
    2. c2(u) = min_{v in unvisited, v != u} (d(u, v) + d(v, destination))
    3. Regret(u) = c2(u) - c1(u)
    4. Score(u) = c1(u) + alpha * Regret(u)
    
    We select the node with the minimum Score.
    Alpha is set to 1.0, making Score(u) = c2(u), which effectively chooses
    the node that leads to the cheapest 2-step path towards the destination.
    """
    n = len(unvisited_nodes)
    
    if n == 0:
        return -1
        
    if n == 1:
        return int(unvisited_nodes[0])
        
    if n == 2:
        # Exact 2-node case: compare both orderings
        n1, n2 = unvisited_nodes
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return int(n1 if cost1 <= cost2 else n2)
        
    # Enforce deadline: if time is running out, fall back to fast 1-step NN
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # 1. Distance from current node to all unvisited candidates
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance matrix among unvisited nodes
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # 3. Distance from each unvisited node to the destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 4. Compute c2(u) = min_{v != u} (d(u, v) + d(v, destination))
    # future_costs[i, j] = d(unvisited[i], unvisited[j]) + d(unvisited[j], destination)
    future_costs = d_sub + d_dest[None, :]
    
    # Set diagonal to inf so u doesn't pick itself as v
    np.fill_diagonal(future_costs, np.inf)
    
    # c2[i] is the min cost to go from unvisited[i] to some other unvisited[j] and then to destination
    c2 = np.min(future_costs, axis=1)
    
    # 5. Compute Regret-2
    # Regret(u) = c2(u) - c1(u)
    # If c2(u) is much larger than c1(u), it means going to u forces us to take a long detour later.
    # We want to minimize the total cost.
    # Score(u) = c1(u) + alpha * (c2(u) - c1(u))
    # Let alpha = 1.0 for pure 2-step lookahead.
    # Score(u) = c2(u)
    
    # However, pure c2 might be too aggressive. 
    # A common variant is Score(u) = c1(u) + 0.5 * (c2(u) - c1(u)) = 0.5 * (c1(u) + c2(u))
    # This balances immediate and future.
    
    # Let's use alpha = 0.5 for a balanced approach.
    alpha = 0.5
    scores = d_curr + alpha * (c2 - d_curr)
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
