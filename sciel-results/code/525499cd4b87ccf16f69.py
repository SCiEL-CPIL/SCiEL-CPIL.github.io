"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a Minimum Spanning Tree (MST) neighbor heuristic.
    
    This approach considers the global structure of the remaining unvisited nodes
    plus the current node. It computes the Minimum Spanning Tree of this subset
    and selects the neighbor of the current node within that MST. This tends to
    avoid long, costly edges that would disconnect the graph or force expensive
    returns later, as MST edges are the most cost-effective connections.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        n1, n2 = int(unvisited_nodes[0]), int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return n1 if cost1 <= cost2 else n2

    # Construct the set of nodes to consider: current + unvisited
    # We need indices into the original distance_matrix
    nodes = np.concatenate(([current_node], unvisited_nodes))
    n_nodes = len(nodes)
    
    # Extract the sub-matrix for these nodes
    # dist_sub[i, j] is distance between nodes[i] and nodes[j]
    dist_sub = distance_matrix[np.ix_(nodes, nodes)]
    
    # Prim's Algorithm to find MST edges
    # in_mst[i] is True if node i is in the MST
    # mst_parent[i] is the parent of node i in the MST
    in_mst = np.zeros(n_nodes, dtype=bool)
    mst_parent = np.full(n_nodes, -1, dtype=int)
    
    # Start from node 0 (which is current_node)
    in_mst[0] = True
    
    # min_edge_to_mst[i] is the minimum distance from node i to any node in the MST
    # min_edge_parent[i] is the node in the MST that provides this minimum distance
    min_edge_to_mst = dist_sub[0].copy()
    min_edge_parent = np.zeros(n_nodes, dtype=int) # All connected to 0 initially
    
    for _ in range(n_nodes - 1):
        # Find the node not in MST with the minimum edge to the MST
        # Mask out nodes already in MST
        min_edge_to_mst[in_mst] = np.inf
        
        # Find index of minimum
        next_node_idx = np.argmin(min_edge_to_mst)
        
        # Add this node to MST
        in_mst[next_node_idx] = True
        mst_parent[next_node_idx] = min_edge_parent[next_node_idx]
        
        # Update min_edge_to_mst for all other nodes
        # For each node j not in MST, check if dist(next_node, j) < min_edge_to_mst[j]
        for j in range(n_nodes):
            if not in_mst[j]:
                if dist_sub[next_node_idx, j] < min_edge_to_mst[j]:
                    min_edge_to_mst[j] = dist_sub[next_node_idx, j]
                    min_edge_parent[j] = next_node_idx

    # Now, find the neighbor(s) of current_node (index 0) in the MST
    # The neighbors of 0 are those nodes i such that mst_parent[i] == 0
    neighbors_in_mst = []
    for i in range(1, n_nodes):
        if mst_parent[i] == 0:
            neighbors_in_mst.append(i)
            
    if not neighbors_in_mst:
        # Fallback: This shouldn't happen if n_nodes > 1, but just in case
        # Pick nearest neighbor
        dists = dist_sub[0, 1:]
        best_local_idx = np.argmin(dists)
        return int(nodes[best_local_idx + 1])
        
    # If there are multiple neighbors in the MST, pick the one with the smallest distance
    # dist_sub[0, i] is the distance from current_node to nodes[i]
    best_neighbor_idx = neighbors_in_mst[0]
    min_dist = dist_sub[0, best_neighbor_idx]
    for i in neighbors_in_mst[1:]:
        d = dist_sub[0, i]
        if d < min_dist:
            min_dist = d
            best_neighbor_idx = i
            
    return int(nodes[best_neighbor_idx])
# EVOLVE-BLOCK-END
