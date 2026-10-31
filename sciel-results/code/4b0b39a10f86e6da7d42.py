# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in a TSP tour using a heuristic that combines
    immediate distance with an MST-based residual estimate.
    
    The heuristic assumes that visiting a node that was connected via a long edge
    in the MST of the unvisited set reduces the residual connectivity cost more
    significantly than visiting a central node. Thus, we reward candidates with
    larger incident edge weights in the MST.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # 1. Compute MST of the current unvisited set
    # We need a dense matrix for scipy's minimum_spanning_tree if n is small
    # or use sparse. n <= 50, so dense is fine.
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    try:
        mst = minimum_spanning_tree(sub_matrix)
        # mst is a sparse matrix
        # We need to find, for each node in unvisited_nodes, the minimum weight of 
        # edges incident to it in the MST.
        
        # Get indices and data from the sparse MST
        # Note: MST is undirected, so scipy might store both directions or just one?
        # minimum_spanning_tree returns a directed graph representation of the undirected tree?
        # Actually, it returns a sparse matrix. For undirected input, it's symmetric?
        # Let's inspect.
        
        # Convert to COO for easier access
        mst_coo = mst.tocoo()
        rows = mst_coo.row
        cols = mst_coo.col
        data = mst_coo.data
        
        # Map global node indices to local indices in unvisited_nodes
        # unvisited_nodes is an array of global indices.
        # We need to know the local index (0 to n_unvisited-1) for each global node.
        # Since sub_matrix is indexed by unvisited_nodes, the local indices are 0..n_unvisited-1
        # corresponding to unvisited_nodes[0], unvisited_nodes[1], etc.
        
        # Initialize min incident edge weight for each local node to infinity
        min_incident_weight = np.full(n_unvisited, float('inf'))
        
        for r, c, w in zip(rows, cols, data):
            # r and c are local indices
            if w < min_incident_weight[r]:
                min_incident_weight[r] = w
            if w < min_incident_weight[c]:
                min_incident_weight[c] = w
        
        # For nodes not in the MST (isolated? shouldn't happen if connected), 
        # or if n_unvisited == 1 (handled), min_incident_weight might be inf.
        # If a node is a leaf, it has 1 edge. If internal, >1.
        # We take the MIN incident weight.
        
        # If a node has no edges in MST (should not happen for connected graph with n>1),
        # set weight to 0.
        min_incident_weight[~np.isfinite(min_incident_weight)] = 0.0
        
    except Exception:
        # Fallback if MST fails
        min_incident_weight = np.zeros(n_unvisited)

    # 2. Select best candidate
    best_score = float('inf')
    best_node = -1
    
    # Heuristic: Score = dist(current, u) + dist(u, destination) - lambda * min_incident_weight[u]
    # We use a negative lambda to reward nodes with high incident edge weights (leaves/outliers)
    # because removing them from the MST reduces the total span more effectively 
    # (conceptually) or leaves a more compact remaining set?
    # Actually, if u is a leaf with edge w, MST(remainder) = MST(total) - w.
    # So residual cost is lower for larger w.
    # So we want to MINIMIZE (dist_current + dist_dest + Residual).
    # Residual ~ MST_total - w_u.
    # Score ~ dist_current + dist_dest - w_u.
    # So yes, subtracting w_u is correct to prefer large w_u.
    
    lambda_val = 1.0  # Scale factor. 1.0 means full weight of the edge is subtracted.
    
    for i in range(n_unvisited):
        candidate = unvisited_nodes[i]
        
        immediate_dist = distance_matrix[current_node, candidate]
        final_dist = distance_matrix[candidate, destination_node]
        residual_bonus = min_incident_weight[i]
        
        # The residual cost of the remaining tour is approximated by MST_total - residual_bonus.
        # Since MST_total is constant for all candidates, we can ignore it in the comparison.
        # We minimize: immediate_dist + final_dist - residual_bonus
        
        total_score = immediate_dist + final_dist - lambda_val * residual_bonus
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate
            
    return int(best_node)
# EVOLVE-BLOCK-END
