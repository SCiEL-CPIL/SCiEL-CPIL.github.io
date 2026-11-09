import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    best_score = np.inf
    best_node = -1
    
    # Precompute distances from current and destination to all unvisited
    dist_current = distance_matrix[current_node, unvisited_nodes]
    dist_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # We will iterate through candidates
    for i, c in enumerate(unvisited_nodes):
        # R is unvisited without c
        r_indices = [j for j in range(n_unvisited) if j != i]
        r_nodes = unvisited_nodes[r_indices]
        
        # V_rem = R U {destination_node}
        # Note: destination_node might be in unvisited_nodes? 
        # The problem statement says "unvisited_nodes". 
        # If destination_node is the final node, it should be in unvisited_nodes.
        # If destination_node is the start node, it is NOT in unvisited_nodes.
        # The current program handles n_unvisited == 1 by returning that node.
        # This implies destination_node IS in unvisited_nodes at the end.
        # So destination_node is part of the set of nodes to be visited.
        
        # Construct V_rem nodes
        # r_nodes does not contain c.
        # Does r_nodes contain destination_node?
        # If i is the index of destination_node, then r_nodes does NOT contain destination_node.
        # We need V_rem = (Unvisited \ {c}).
        # So V_rem = r_nodes.
        # Wait, the 1-tree bound requires the root to be c.
        # The set of nodes in the tour is V' = {c} U V_rem.
        # V_rem = Unvisited \ {c}.
        # So we compute MST of V_rem.
        
        n_rem = len(r_nodes)
        if n_rem == 0:
            # Only c and dest? If n_unvisited == 1, handled above.
            # So n_rem >= 1.
            pass

        # Extract submatrix for V_rem
        sub_dist = distance_matrix[np.ix_(r_nodes, r_nodes)]
        
        # Compute MST of V_rem
        # Use Prim's algorithm as before, but on V_rem
        n_r = n_rem
        if n_r == 0:
            mst_w = 0
        else:
            visited = np.zeros(n_r, dtype=bool)
            min_edge = np.full(n_r, np.inf)
            min_edge[0] = 0
            mst_w = 0
            
            for _ in range(n_r):
                temp = min_edge.copy()
                temp[visited] = np.inf
                u = np.argmin(temp)
                
                if min_edge[u] == np.inf:
                    break
                    
                mst_w += min_edge[u]
                visited[u] = True
                
                min_edge = np.where(visited, min_edge, np.minimum(min_edge, sub_dist[u]))
        
        # Find two cheapest edges from c to V_rem
        dist_c_to_rem = distance_matrix[c, r_nodes]
        
        if n_rem == 1:
            # Only one node in V_rem
            # The "two cheapest" edges must both go to the same node?
            # In a 1-tree, the root has degree 2.
            # If V_rem has 1 node, the tour is c -> v -> c.
            # The 1-tree bound: MST({v}) = 0.
            # Two cheapest edges from c to {v} are d(c,v) and d(c,v).
            # LB = 0 + d(c,v) + d(c,v) = 2*d(c,v).
            # Path cost = LB - d(c, dest).
            # If dest is that node v, Path cost = 2*d(c,v) - d(c,v) = d(c,v).
            # Correct.
            d1 = dist_c_to_rem[0]
            d2 = dist_c_to_rem[0]
        else:
            # Find two smallest
            # Sort or use argpartition
            # For small N, sorting is fine
            sorted_dists = np.sort(dist_c_to_rem)
            d1 = sorted_dists[0]
            d2 = sorted_dists[1]
            
        lb_tour = mst_w + d1 + d2
        # Path cost estimate
        # The tour on V' = {c} U V_rem has length >= lb_tour.
        # The path from c to dest (where dest is in V_rem) has length >= lb_tour - d(c, dest).
        # Note: dest is in V_rem.
        d_c_dest = distance_matrix[c, destination_node]
        
        score = dist_current[i] + lb_tour - d_c_dest
        
        if score < best_score:
            best_score = score
            best_node = c
            
    return int(best_node)
