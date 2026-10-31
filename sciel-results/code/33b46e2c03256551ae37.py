"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse.csgraph as csg

_START_TIME = time.time()
_TOTAL_LIMIT = 150.0
_MARGIN = 5.0

def _is_deadline_passed():
    """Check if we are close to the time limit."""
    return (time.time() - _START_TIME) > (_TOTAL_LIMIT - _MARGIN)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Hybrid MST-Aware and 2-Step Lookahead Heuristic.
    
    Combines the structural insight of MST-leaf-peeling (from inspiration 2) with 
    the local refinement of 2-step lookahead (from current program).
    
    Score(u) = d(current, u) + alpha * d(u, nn(u, U\u2216{u})) + beta * penalty_degree(u)
    
    Where:
    - d(current, u) is the immediate cost.
    - d(u, nn(u, U\u2216{u})) is the 2-step lookahead cost (distance to nearest neighbor in remaining set).
    - penalty_degree(u) is based on the degree of u in the MST of the unvisited set.
      High degree nodes are central to the remaining graph structure; removing them
      might fragment the graph or force long connections later. We prefer to peel
      off leaves (low degree) or nodes that are not critical hubs, unless the immediate
      cost is very low.
      
    We use a soft penalty for high MST degree to encourage peeling leaves, while
    retaining the 2-step lookahead to ensure local optimality.
    """
    n_unvisited = len(unvisited_nodes)
    
    # Base cases
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Enforce deadline: if time is running out, fall back to fast 1-step NN
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # Extract distances from current node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Extract the sub-matrix of distances between unvisited nodes
    unvisited_indices = unvisited_nodes
    D_unvisited = distance_matrix[np.ix_(unvisited_indices, unvisited_indices)].copy()
    
    # --- Component 1: 2-Step Lookahead (from Current Program) ---
    # Set diagonal to infinity to exclude self
    np.fill_diagonal(D_unvisited, np.inf)
    min_dist_to_other = np.min(D_unvisited, axis=1)
    
    # --- Component 2: MST Degree Penalty (from Inspiration 2) ---
    # Compute MST of the unvisited subgraph to determine structural importance
    # We use a copy because csg.minimum_spanning_tree might modify or we just want to be safe
    D_unvisited_for_mst = D_unvisited.copy()
    # Restore diagonal for MST calculation (MST doesn't care about self-loops but standard input is full matrix)
    # Actually, minimum_spanning_tree usually ignores diagonal or treats it as 0. 
    # Let's set diagonal to 0 to be safe for graph algorithms that might check connectivity via self? No, self-loops are ignored.
    # The previous D_unvisited has inf on diagonal. Let's create a clean matrix for MST.
    D_clean = distance_matrix[np.ix_(unvisited_indices, unvisited_indices)]
    
    try:
        mst = csg.minimum_spanning_tree(D_clean)
        mst_coo = mst.tocoo()
        
        degrees = np.zeros(n_unvisited, dtype=int)
        for i, j in zip(mst_coo.row, mst_coo.col):
            degrees[i] += 1
            degrees[j] += 1
            
        # Normalize degrees to [0, 1] range roughly
        # Max degree in a tree with N nodes is N-1, but usually small.
        # We want to penalize high degree.
        # If a node is a leaf (degree 1), penalty is 0.
        # If a node is a hub (high degree), penalty is higher.
        # Let's map degree to a penalty.
        # Penalty = (degree - 1) * scale_factor
        # Or simply use the degree itself as a penalty term, scaled.
        
        # The "leaf peeling" idea suggests we should prefer nodes with low degree in the MST.
        # However, we must balance this with distance.
        # Let's use a soft penalty: penalty = 0.05 * max(0, degrees - 1)
        # This encourages picking leaves (degree 1) over hubs (degree > 1).
        
    except Exception:
        # Fallback if MST fails
        degrees = np.ones(n_unvisited, dtype=int)
        
    mst_penalty = 0.05 * np.maximum(0, degrees - 1)
    
    # --- Combine Scores ---
    # Score = Immediate Cost + Lookahead Cost + MST Penalty
    # The lookahead cost is already a distance, so it's on the same scale.
    # The MST penalty is a small additive term to break ties or discourage hubs.
    
    scores = dists_from_current + min_dist_to_other + mst_penalty
    
    # Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
