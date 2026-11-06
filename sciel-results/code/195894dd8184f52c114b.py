"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using an MST-based structural heuristic.
    
    This approach computes the Minimum Spanning Tree (MST) of the subgraph 
    containing the current node and all unvisited nodes. The next node is 
    chosen as the neighbor of the current node in this MST. If the current 
    node has multiple neighbors in the MST, the one with the smallest 
    Euclidean distance is selected. This leverages global connectivity 
    information to avoid local greedy traps.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return destination_node
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    # Combine current node and unvisited nodes for MST calculation
    # Note: We do NOT include destination_node in the MST because we are 
    # constructing the path forward, and the destination is the final step.
    # Including it might bias the MST towards the end prematurely.
    nodes = np.concatenate(([current_node], unvisited_nodes))
    n = len(nodes)
    
    # Extract the relevant distance matrix sub-block
    # dist_sub[i, j] is distance between nodes[i] and nodes[j]
    dist_sub = distance_matrix[np.ix_(nodes, nodes)]
    
    # Prim's Algorithm to find MST
    # in_mst[i] indicates if node i is in the MST
    # min_dist[i] is the min distance to connect node i to the MST
    # parent[i] is the node in MST that connects to i
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    
    # Start from current_node (index 0 in 'nodes')
    min_dist[0] = 0.0
    in_mst[0] = True
    
    for _ in range(n - 1):
        # Find the node with min distance not in MST
        # Mask out nodes already in MST
        valid_dist = np.where(in_mst, np.inf, min_dist)
        next_idx = np.argmin(valid_dist)
        
        if min_dist[next_idx] == np.inf:
            break # Disconnected graph? Should not happen for Euclidean
        
        in_mst[next_idx] = True
        
        # Update distances for neighbors
        # dist_sub[next_idx, :] gives distances from next_idx to all others
        # We only update if the new distance is smaller
        for j in range(n):
            if not in_mst[j] and dist_sub[next_idx, j] < min_dist[j]:
                min_dist[j] = dist_sub[next_idx, j]
                parent[j] = next_idx
        
    # Identify neighbors of current_node (index 0) in the MST
    # A node j is a neighbor of 0 if parent[j] == 0 OR parent[0] == j
    # Since 0 is the root, parent[0] is -1. So we just look for parent[j] == 0.
    neighbors_of_current = []
    for j in range(1, n):
        if parent[j] == 0:
            neighbors_of_current.append(j)
    
    # If no neighbors found (should not happen if n>1), fallback to nearest
    if not neighbors_of_current:
        # Fallback: Nearest Neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])
        
    if len(neighbors_of_current) == 1:
        # Map index in 'nodes' back to original node id
        # nodes[0] is current, nodes[1:] are unvisited
        # neighbor index j corresponds to unvisited_nodes[j-1]
        neighbor_idx_in_unvisited = neighbors_of_current[0] - 1
        return int(unvisited_nodes[neighbor_idx_in_unvisited])
    else:
        # Multiple neighbors in MST: pick the one with smallest distance to current
        best_neighbor_idx = None
        best_dist = np.inf
        for j in neighbors_of_current:
            d = dist_sub[0, j]
            if d < best_dist:
                best_dist = d
                best_neighbor_idx = j
        
        neighbor_idx_in_unvisited = best_neighbor_idx - 1
        return int(unvisited_nodes[neighbor_idx_in_unvisited])

# EVOLVE-BLOCK-END
