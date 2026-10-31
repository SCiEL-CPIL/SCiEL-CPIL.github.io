# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 145.0  # Leave 5s margin for 150s limit

def _is_timeout():
    return (time.time() - _START_TIME) > _TIMEOUT_LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in a TSP tour using a 2-step lookahead heuristic.
    
    Heuristic: Score(u) = d(current, u) + lambda * min_{v in U\{u}} d(u, v)
    This avoids choosing a node that is cheap to reach but forces a very expensive
    subsequent move.
    """
    if _is_timeout():
        # Fallback to nearest neighbor if we are running out of time
        if len(unvisited_nodes) == 0:
            return int(current_node)
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        n1, n2 = unvisited_nodes
        # Cost if we go to n1 first: current -> n1 -> n2 -> destination
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        # Cost if we go to n2 first: current -> n2 -> n1 -> destination
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return int(n1 if cost1 <= cost2 else n2)

    # Vectorized 2-step lookahead
    # 1. Distance from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance matrix between unvisited nodes
    # This is an NxN matrix where N is n_unvisited
    d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # 3. For each candidate u (row i), find the minimum distance to any other unvisited node v (j != i)
    # We set the diagonal to infinity so that min() ignores the distance to self
    d_unvisited_no_self = d_unvisited.copy()
    np.fill_diagonal(d_unvisited_no_self, np.inf)
    
    # min_next_cost[i] is the distance from unvisited_nodes[i] to its nearest neighbor in unvisited_nodes
    min_next_cost = np.min(d_unvisited_no_self, axis=1)
    
    # 4. Combine scores
    # lambda balances immediate cost vs future cost. 
    # A value of 1.0 is a standard choice for "greedy with lookahead".
    # We can tune this, but 1.0 is a robust default.
    lambda_weight = 1.0
    
    scores = dist_from_current + lambda_weight * min_next_cost
    
    # 5. Select the candidate with the lowest score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
