import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if len(unvisited_nodes) == 0:
        return -1 # Should not happen
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    # Precompute distances from current to all unvisited
    dists = distance_matrix[current_node, unvisited_nodes]
    
    # Dist from destination to current (fixed)
    d_dest_cur = distance_matrix[destination_node, current_node]
    
    # Dist from destination to all unvisited
    d_dest_unv = distance_matrix[destination_node, unvisited_nodes]
    
    scores = np.zeros(len(unvisited_nodes))
    
    # Lambda for turn penalty
    lambda_turn = 0.5
    
    for i, u in enumerate(unvisited_nodes):
        d_cur_u = dists[i]
        d_dest_u = d_dest_unv[i]
        
        # Calculate cos(beta) where beta is angle at current in triangle (dest, cur, u)
        # Law of Cosines: d_dest_u^2 = d_dest_cur^2 + d_cur_u^2 - 2 * d_dest_cur * d_cur_u * cos(beta)
        # cos(beta) = (d_dest_cur^2 + d_cur_u^2 - d_dest_u^2) / (2 * d_dest_cur * d_cur_u)
        
        denom = 2 * d_dest_cur * d_cur_u
        if denom < 1e-10:
            # Degenerate case
            cos_beta = 1.0 # Assume backtrack or straight, penalize to be safe? 
            # If d_cur_u is 0, we are at u? No, u is unvisited.
            # If d_dest_cur is 0, start==current.
            pass
        else:
            cos_beta = (d_dest_cur**2 + d_cur_u**2 - d_dest_u**2) / denom
            # Clamp to [-1, 1]
            cos_beta = np.clip(cos_beta, -1.0, 1.0)
            
        # Penalty: 0 for straight (cos=-1), 2 for backtrack (cos=1)
        turn_penalty = 1.0 + cos_beta
        
        scores[i] = d_cur_u + lambda_turn * turn_penalty
        
    # Pick min score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
