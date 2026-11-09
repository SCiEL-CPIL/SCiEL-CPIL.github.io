import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1 

    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # 1. Get distances from current to all unvisited
    direct_costs = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Get distances from all unvisited nodes to the destination
    dest_costs = distance_matrix[unvisited_nodes, destination_node]
    
    # 3. Get sub-matrix of distances among unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # 4. Compute lookahead costs with destination bias
    # For each candidate i, we estimate the cost of moving to i and then to the best next node j.
    # The cost of j is biased by its distance to the destination to help close the tour.
    # Cost(i) = dist(current, i) + min_{j != i} (dist(i, j) + lambda * dist(j, destination))
    
    # lambda_weight controls the strength of the destination bias
    lambda_weight = 1.0
    
    # Create the lookahead matrix: M[i, j] = dist(i, j) + lambda * dist(j, destination)
    # dist(j, destination) is dest_costs[j]
    # We add this to each row i of sub_matrix, but the value depends on column j.
    # So we broadcast dest_costs along the rows.
    lookahead_matrix = sub_matrix + lambda_weight * dest_costs[np.newaxis, :]
    
    # Prevent picking the candidate itself as the next node
    np.fill_diagonal(lookahead_matrix, np.inf)
    
    # Find the minimum lookahead cost for each candidate i
    lookahead_costs = np.min(lookahead_matrix, axis=1)
    
    # 5. Combine direct and lookahead costs
    total_costs = direct_costs + lookahead_costs
    
    # 6. Select the best candidate
    best_local_idx = np.argmin(total_costs)
    
    return int(unvisited_nodes[best_local_idx])
