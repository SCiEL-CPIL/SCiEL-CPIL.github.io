# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix


def _mst_min_edge(nodes, distance_matrix):
    """
    Compute the minimum edge weight in the MST of the given nodes.
    This serves as a lightweight proxy for the residual tour complexity.
    """
    n = len(nodes)
    if n < 2:
        return 0.0
    
    sub_matrix = distance_matrix[np.ix_(nodes, nodes)]
    
    try:
        mst = minimum_spanning_tree(sub_matrix)
        if mst.nnz == 0:
            return 0.0
        return float(np.min(mst.data))
    except Exception:
        return float('inf')


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in a TSP tour using a heuristic that combines
    immediate distance with an MST-min-edge-based estimate of the residual tour cost.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Pre-compute the minimum edge weight in the MST of all unvisited nodes
    # This is a cheap proxy for how "spread out" the remaining nodes are.
    # A smaller min-edge suggests the remaining nodes are tightly clustered,
    # making it cheaper to visit them.
    mst_min_edge_all = _mst_min_edge(unvisited_nodes, distance_matrix)
    
    best_score = float('inf')
    best_node = -1
    
    # Pre-compute distances from current and destination to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_from_destination = distance_matrix[destination_node, unvisited_nodes]
    
    for i in range(n_unvisited):
        candidate = unvisited_nodes[i]
        
        # Immediate cost: distance from current to candidate
        immediate_dist = dist_from_current[i]
        
        # Residual cost estimate:
        # After visiting candidate, we need to visit the remaining n-1 nodes and return to destination.
        # The residual cost is approximated by:
        #   - The cost to connect candidate to the remaining cluster: min edge from candidate to remaining
        #   - The MST min edge of the remaining set (proxy for residual complexity)
        #   - The cost to connect the remaining cluster to destination: min edge from destination to remaining
        
        remaining_indices = np.concatenate([np.arange(i), np.arange(i+1, n_unvisited)])
        remaining_nodes = unvisited_nodes[remaining_indices]
        
        if len(remaining_nodes) == 0:
            # Should not happen since n_unvisited >= 2 here
            residual_est = dist_from_destination[i]
        else:
            # Min edge from candidate to remaining nodes
            min_cand_to_rem = np.min(distance_matrix[candidate, remaining_nodes])
            
            # Min edge from destination to remaining nodes
            min_dest_to_rem = np.min(distance_matrix[destination_node, remaining_nodes])
            
            # MST min edge of remaining nodes (lightweight residual proxy)
            mst_min_edge_rem = _mst_min_edge(remaining_nodes, distance_matrix)
            
            # Combine: the residual tour must at least span the remaining nodes (MST min edge)
            # plus connect the entry (candidate) and exit (destination) to the cluster
            residual_est = min_cand_to_rem + min_dest_to_rem + mst_min_edge_rem
        
        total_score = immediate_dist + residual_est
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate
            
    return int(best_node)
# EVOLVE-BLOCK-END
