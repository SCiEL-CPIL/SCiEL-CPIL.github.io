"""Direct heuristic for TSP next-node selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # safety margin before 150s hard kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction.
    
    Structural Change: Global 2-Step Lookahead.
    Instead of a simple blend of distances, we evaluate each candidate node $i$
    by the cost of the path $current \to i \to j \to destination$, where $j$ is
    the best possible next node after $i$ (excluding $i$ itself).
    
    Score(i) = d(current, i) + min_{j in Unvisited \ {i}} (d(i, j) + d(j, destination))
    
    This heuristic:
    1. Considers the immediate cost to reach the candidate.
    2. Considers the cost to leave the candidate to the next best node.
    3. Considers the cost to close the tour from that next node.
    
    This effectively performs a 2-step lookahead for the entire tour construction,
    which is computationally feasible for N=50 using vectorized NumPy operations.
    """
    # Time guard
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        # For two nodes left, pick the one that minimizes total edge cost to close the tour
        u, v = unvisited_nodes
        # Option 1: go to u then v: cost = d(current,u) + d(u,v) + d(v,dest)
        cost_u = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
        # Option 2: go to v then u: cost = d(current,v) + d(v,u) + d(u,dest)
        cost_v = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
        return int(unvisited_nodes[0] if cost_u <= cost_v else unvisited_nodes[1])
    
    # Vectorized 2-Step Lookahead
    # d_curr: distances from current_node to each unvisited node
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # d_dest: distances from each unvisited node to destination_node
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # D_sub: distance matrix between all unvisited nodes
    # Shape: (n_unvisited, n_unvisited)
    D_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # We want to compute M[i] = min_{j != i} (D_sub[i, j] + d_dest[j])
    # Create a matrix of costs: Cost[i, j] = D_sub[i, j] + d_dest[j]
    # d_dest is shape (n_unvisited,), so it broadcasts along rows.
    Costs = D_sub + d_dest
    
    # Set diagonal to infinity to exclude j == i
    np.fill_diagonal(Costs, np.inf)
    
    # Find the minimum cost for each candidate i (row i)
    best_next_costs = np.min(Costs, axis=1)
    
    # Total score for each candidate i
    scores = d_curr + best_next_costs
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
