"""Rewritten program: TSP constructive heuristic using MST-leaf-peeling strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse.csgraph as csg

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
    
    Strategy: MST-Leaf-Peeling with Destination Awareness.
    
    1. Compute the Minimum Spanning Tree (MST) of the unvisited nodes.
    2. Identify the degree of each node in this MST.
    3. Score candidates based on:
       - Immediate distance from current node.
       - Penalty for high degree in MST (prefer leaves to "peel" them off).
       - Proximity to destination (weighted more heavily near the end of the tour).
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
    
    # 2. Distance from each unvisited node to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 3. Compute MST of the unvisited subgraph
    # Extract the submatrix for unvisited nodes
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Use scipy to find MST
    # csg.minimum_spanning_tree expects a weighted graph
    # We can pass the distance matrix directly as a dense matrix? 
    # scipy.sparse.csgraph.minimum_spanning_tree works on sparse or dense.
    # For 50 nodes, dense is fine.
    
    try:
        mst = csg.minimum_spanning_tree(d_sub)
        # mst is a sparse matrix. Convert to COO to get edges
        mst_coo = mst.tocoo()
        
        # Initialize degree array
        degrees = np.zeros(n, dtype=int)
        
        # Count degrees from MST edges
        for i, j in zip(mst_coo.row, mst_coo.col):
            degrees[i] += 1
            degrees[j] += 1
            
        # Calculate median edge length in the subgraph for scaling
        # We need the actual weights of the MST edges to scale the penalty
        # mst_coo.data contains the weights
        if len(mst_coo.data) > 0:
            median_mst_edge = np.median(mst_coo.data)
        else:
            median_mst_edge = 1.0
            
    except Exception:
        # Fallback if MST fails (e.g., disconnected, though TSP instances are usually connected)
        degrees = np.ones(n, dtype=int)
        median_mst_edge = np.median(d_curr) if n > 0 else 1.0

    # 4. Compute Score with smooth surrogates
    # We want to minimize the score.
    # Base cost: d_curr
    # Penalty: Smooth exponential penalty for high degree (prefer leaves)
    # Destination proximity: Stronger ramp as n decreases
    # Regret: How uniquely well-positioned is this node?
    
    # Smooth degree penalty: exp(alpha * (degree - min_degree))
    deg_min = np.min(degrees)
    deg_range = np.max(degrees) - deg_min
    if deg_range > 0:
        alpha = 3.0 / (deg_range * (median_mst_edge + 1e-9))
        deg_penalty = np.exp(alpha * (degrees - deg_min)) - 1.0
    else:
        deg_penalty = np.zeros(n)
    
    # Destination weight: stronger ramp
    # At n=50: 0.1+0.04=0.14, at n=10: 0.1+0.2=0.3, at n=1: 0.1+2.0=2.1
    dest_weight = 0.1 + 2.0 / n
    
    # Regret: for each u, regret(u) = min_{v != u} [d(curr,v) + d(v,u)] - d(curr,u)
    # High regret = uniquely well-positioned, should pick now
    # Compute pairwise: cost of going curr->v->u vs curr->u
    d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    # For each pair (i,j): d(curr,i) + d(i,j) - d(curr,j)
    # regret[j] = min over i != j of [d_curr[i] + d_unvisited[i,j]] - d_curr[j]
    # Build matrix: M[i,j] = d_curr[i] + d_unvisited[i,j]
    M = d_curr[:, np.newaxis] + d_unvisited  # M[i,j] = d(curr,i) + d(i,j)
    # For each column j, find min over i != j
    # Set diagonal to inf to exclude self
    np.fill_diagonal(M, np.inf)
    min_through = np.min(M, axis=0)  # min over i for each j
    regret = min_through - d_curr  # positive means u is better direct
    
    # Clip negative regret to 0 (if going through someone is cheaper, no regret)
    regret = np.maximum(regret, 0)
    
    # Degree penalty weight
    deg_weight = 0.15 * median_mst_edge
    
    # Regret weight: scale by median edge to be comparable
    regret_weight = 0.2
    
    # Immediate cost weight
    curr_weight = 1.0
    
    scores = (curr_weight * d_curr 
              + deg_weight * deg_penalty 
              + dest_weight * d_dest
              - regret_weight * regret)
    
    # Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
