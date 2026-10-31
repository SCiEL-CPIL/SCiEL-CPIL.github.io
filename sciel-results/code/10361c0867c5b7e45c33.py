# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix


def _compute_mst_min_edge(nodes, distance_matrix):
    """
    Compute the minimum edge weight in the MST of the given nodes.
    This serves as a lightweight estimate of the residual tour cost per unit.
    """
    n = len(nodes)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0
    
    # Extract submatrix for unvisited nodes
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
    immediate distance with an MST-based estimate of the residual tour cost.
    Uses the minimum edge weight in the MST of remaining nodes as a proxy for residual cost.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    best_score = float('inf')
    best_node = -1
    
    for i in range(n_unvisited):
        candidate = unvisited_nodes[i]
        
        # Immediate cost: distance from current to candidate
        immediate_dist = distance_matrix[current_node, candidate]
        
        # Residual cost estimate using MST minimum edge weight
        remaining_indices = [j for j in range(n_unvisited) if j != i]
        remaining_nodes = unvisited_nodes[remaining_indices]
        
        if len(remaining_nodes) == 0:
            residual_est = 0.0
        else:
            # Use minimum edge in MST as a characteristic length scale for remaining sub-tour
            min_edge = _compute_mst_min_edge(remaining_nodes, distance_matrix)
            
            # The residual tour involves visiting all remaining nodes and returning to destination.
            # Estimate: (number of remaining nodes) * min_edge + connection costs
            # This scales the characteristic edge length by the number of nodes to visit.
            num_remaining = len(remaining_nodes)
            
            # Connection cost from candidate to nearest remaining node
            min_dist_cand_rem = np.min(distance_matrix[candidate, remaining_nodes])
            # Connection cost from destination to nearest remaining node  
            min_dist_dest_rem = np.min(distance_matrix[destination_node, remaining_nodes])
            
            # Total residual estimate:
            # - The sub-tour through remaining nodes has cost proportional to num_remaining * min_edge
            # - Plus connections from candidate to the cluster and from cluster to destination
            residual_est = num_remaining * min_edge + min_dist_cand_rem + min_dist_dest_rem
        
        total_score = immediate_dist + residual_est
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate
            
    return int(best_node)
# EVOLVE-BLOCK-END
