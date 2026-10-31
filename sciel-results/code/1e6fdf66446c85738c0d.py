"""Rewritten program: TSP constructive heuristic using Minimax Bottleneck Strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline enforcement
_start_time = time.time()
_TOTAL_LIMIT = 150.0  # Evaluator limit
_MARGIN = 5.0         # Safety margin

def _is_deadline_passed():
    """Check if we are close to the time limit."""
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Minimax Bottleneck Heuristic.
    For each candidate node u in unvisited_nodes, we calculate:
    1. Immediate cost: d(current, u)
    2. Bottleneck cost: max_{v in unvisited union {destination}} d(u, v)
    
    The total score is a weighted sum:
    score(u) = alpha * d(current, u) + (1 - alpha) * bottleneck(u)
    
    This structural change aims to prevent long edges by ensuring that
    the chosen node is not too far from any remaining node, thereby
    minimizing the maximum edge length in the emerging tour structure.
    """
    n = len(unvisited_nodes)
    
    if n == 0:
        return -1
        
    if n == 1:
        return int(unvisited_nodes[0])
        
    # Enforce deadline: if time is running out, fall back to fast 1-step NN
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # 1. Distance from current node to all unvisited candidates
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance matrix among unvisited nodes
    # d_sub[i, j] is distance between unvisited_nodes[i] and unvisited_nodes[j]
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # 3. Distance from each unvisited node to the destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 4. Calculate Bottleneck Cost for each candidate u
    # We want max_{v in unvisited U {dest}} d(u, v)
    # Create a matrix D where columns are distances to all other unvisited nodes + destination
    # d_sub is (n, n), d_dest is (n,)
    # Stack d_dest as a column to d_sub
    d_all = np.hstack((d_sub, d_dest[:, np.newaxis]))
    
    # The max distance from u to any other unvisited node or the destination
    # Note: d_sub[u, u] is 0, so it won't affect the max unless n=1 (handled above)
    bottleneck_costs = np.max(d_all, axis=1)
    
    # 5. Hybrid Score
    # Balance between local proximity (NN) and global bottleneck reduction
    # Alpha = 0.5 suggests equal weight. Tuning this might be needed, 
    # but 0.5 is a standard structural start for this hybrid.
    alpha = 0.5
    scores = alpha * d_curr + (1 - alpha) * bottleneck_costs
    
    # 6. Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
