"""Rewritten program: TSP constructive heuristic using MST-based residual cost estimation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse.csgraph import minimum_spanning_tree

# Track start time for deadline enforcement
_start_time = time.time()
_TOTAL_LIMIT = 150.0  # Evaluator limit
_MARGIN = 5.0         # Safety margin

def _is_deadline_passed():
    """Check if we are close to the time limit."""
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def _mst_weight(d_sub):
    """Compute the total weight of the MST for a distance submatrix."""
    n = d_sub.shape[0]
    if n <= 1:
        return 0.0
    # Convert to sparse graph for MST computation
    # Use the distance matrix directly
    from scipy.sparse import csr_matrix
    # Create a sparse graph where edge weights are distances
    # For MST, we need to handle the fact that diagonal is 0
    d_copy = d_sub.copy()
    np.fill_diagonal(d_copy, 0)
    # Convert to sparse
    sparse = csr_matrix(d_copy)
    try:
        mst = minimum_spanning_tree(sparse)
        return float(mst.sum())
    except Exception:
        # Fallback: use a simple approximation
        # Sum of minimum incident edge for each node, divided by 2
        min_edges = np.min(d_sub, axis=1)
        return float(np.sum(min_edges) / 2.0)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: MST-based residual cost estimation.
    
    For each candidate u, estimate the total tour cost if we visit u next:
    1. Immediate cost: d(current, u)
    2. Residual cost: MST weight of the remaining unvisited nodes (after removing u)
       This captures the connectivity cost of the remaining subproblem.
    3. Connection cost: d(u, destination) to help close the tour
    
    score(u) = d(current, u) + alpha * MST(remaining_after_u) + beta * d(u, dest)
    
    The MST weight serves as a lower bound on the residual tour cost,
    guiding us to preserve low-cost connectivity for the remaining cities.
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
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # 3. Distance from each unvisited node to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 4. For each candidate u, compute the MST weight of the remaining nodes
    #    remaining_after_u = unvisited_nodes excluding u
    #    This is O(n^3) in the worst case, but n is at most 50
    
    mst_costs = np.zeros(n)
    for i in range(n):
        # Remove node i from the unvisited set
        if n - 1 <= 1:
            mst_costs[i] = 0.0
        else:
            # Create the submatrix excluding node i
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            d_remaining = d_sub[np.ix_(mask, mask)]
            mst_costs[i] = _mst_weight(d_remaining)
    
    # 5. Normalize components to combine them
    # Immediate cost
    curr_min = np.min(d_curr)
    curr_max = np.max(d_curr)
    if curr_max > curr_min:
        curr_norm = (d_curr - curr_min) / (curr_max - curr_min)
    else:
        curr_norm = np.zeros(n)
    
    # MST cost
    mst_min = np.min(mst_costs)
    mst_max = np.max(mst_costs)
    if mst_max > mst_min:
        mst_norm = (mst_costs - mst_min) / (mst_max - mst_min)
    else:
        mst_norm = np.zeros(n)
    
    # Destination proximity
    dest_min = np.min(d_dest)
    dest_max = np.max(d_dest)
    if dest_max > dest_min:
        dest_norm = (d_dest - dest_min) / (dest_max - dest_min)
    else:
        dest_norm = np.zeros(n)
    
    # 6. Combine scores with weights:
    #    We want to MINIMIZE the total score.
    #    - Higher immediate cost -> worse (positive weight)
    #    - Higher MST residual cost -> worse (positive weight, prefer moves that preserve connectivity)
    #    - Higher dest distance -> worse (positive weight, prefer nodes closer to dest)
    
    w_immediate = 1.0
    w_mst = 1.5        # Weight for MST residual cost (strong emphasis on connectivity)
    w_dest = 0.3       # Weight for destination proximity
    
    scores = (w_immediate * curr_norm 
              + w_mst * mst_norm 
              + w_dest * dest_norm)
    
    # 7. Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
