import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Map unvisited indices to 0..K-1 for local matrix operations if needed, 
    # but we can just use the global distance_matrix.
    
    best_score = np.inf
    best_node = -1
    
    # Precompute distances from current and destination to all unvisited
    dist_current = distance_matrix[current_node, unvisited_nodes]
    dist_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # We need to compute MST for subsets of unvisited_nodes.
    # To speed up MST, we can extract the submatrix.
    
    for i, c in enumerate(unvisited_nodes):
        # R is unvisited without c
        if n_unvisited == 2:
            # R has 1 node
            r_idx = 1 - i # The other index
            r_node = unvisited_nodes[r_idx]
            # Path: current -> c -> r_node -> destination
            cost = dist_current[i] + distance_matrix[c, r_node] + distance_matrix[r_node, destination_node]
            # Note: min_c_R is d(c, r_node), min_dest_R is d(dest, r_node), MST_R is 0
            # Formula: d(cur,c) + d(c,r) + d(dest,r) + 0
            if cost < best_score:
                best_score = cost
                best_node = c
            continue
            
        # General case
        # Create mask for R
        # R indices in unvisited_nodes are all j != i
        r_indices = [j for j in range(n_unvisited) if j != i]
        r_nodes = unvisited_nodes[r_indices]
        
        # min_c_R: min distance from c to any node in R
        # We can use distance_matrix[c, r_nodes]
        min_c_R = np.min(distance_matrix[c, r_nodes])
        
        # min_dest_R: min distance from destination to any node in R
        min_dest_R = np.min(dist_dest[r_indices])
        
        # MST of R
        # Extract submatrix for R
        # r_nodes is a numpy array of node IDs
        sub_dist = distance_matrix[np.ix_(r_nodes, r_nodes)]
        
        # Compute MST weight
        # Use scipy for robustness, or manual Prim
        # For N=50, manual Prim in NumPy might be faster than scipy overhead?
        # Let's try scipy first, it's standard.
        # But scipy requires sparse or dense.
        # minimum_spanning_tree expects a graph.
        # We can use scipy.sparse.csgraph.minimum_spanning_tree(sub_dist)
        # But sub_dist is dense.
        
        # Alternative: Implement Prim's with NumPy
        # This is often faster for small N than calling scipy overhead.
        
        # Let's implement a fast Prim's in NumPy
        # Nodes in R are indexed 0..len(r_nodes)-1 in the submatrix
        n_r = len(r_nodes)
        if n_r == 0:
            mst_w = 0
        else:
            # Prim's algorithm
            # visited mask
            visited = np.zeros(n_r, dtype=bool)
            # min edge to connect each node to the tree
            # Initialize with inf, except start node 0
            min_edge = np.full(n_r, np.inf)
            min_edge[0] = 0
            mst_w = 0
            
            for _ in range(n_r):
                # Find unvisited node with min edge
                # Mask visited to inf
                temp = min_edge.copy()
                temp[visited] = np.inf
                u = np.argmin(temp)
                
                if min_edge[u] == np.inf:
                    # Disconnected? Should not happen for TSP
                    break
                    
                mst_w += min_edge[u]
                visited[u] = True
                
                # Update min_edge for neighbors of u
                # dist from u to all unvisited nodes
                # sub_dist[u, :] gives distances from u to all nodes in R
                # We only care about unvisited ones
                # new_min = min(current_min_edge, sub_dist[u, :])
                # But only for unvisited nodes
                
                # Vectorized update
                # We want to update min_edge[j] = min(min_edge[j], sub_dist[u, j]) for j not visited
                # We can do this by taking min of min_edge and sub_dist[u], then masking visited
                # But we must not update visited nodes.
                
                # Create a candidate array
                # candidates = np.minimum(min_edge, sub_dist[u])
                # candidates[visited] = min_edge[visited] # Keep old values for visited (though they are inf or processed)
                # Actually, we just want to update unvisited.
                
                # Efficient way:
                # min_edge = np.where(visited, min_edge, np.minimum(min_edge, sub_dist[u]))
                
                min_edge = np.where(visited, min_edge, np.minimum(min_edge, sub_dist[u]))
            
            # mst_w is the weight
            
        cost = dist_current[i] + min_c_R + min_dest_R + mst_w
        
        if cost < best_score:
            best_score = cost
            best_node = c
            
    return int(best_node)
