"""Direct heuristic for TSP next-node selection with look-ahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # safety margin before 150s hard kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction.
    
    Strategy:
    - For small sets of unvisited nodes (n <= 2), use exact optimal completion.
    - For medium sets (n <= 20), use a heuristic that strongly favors nodes
      close to both the current node and the destination (start node) to
      ensure a short closing edge.
    - For large sets (n > 20), use a 2-step look-ahead heuristic:
      Minimize d(current, u) + min(d(u, v) for v in unvisited, v != u).
      This avoids "dead ends" where the next node is close to current but
      far from all other unvisited nodes.
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
    
    # Threshold for switching strategy
    # We use a stronger destination bias for the last 20 nodes.
    if n_unvisited <= 20:
        # Minimize d(current, u) + d(u, destination)
        # This effectively treats the remaining tour as a path from current to destination
        # through the remaining nodes, but greedily.
        d_curr = distance_matrix[current_node, unvisited_nodes]
        d_dest = distance_matrix[unvisited_nodes, destination_node]
        scores = d_curr + d_dest
        best_idx = np.argmin(scores)
        return int(unvisited_nodes[best_idx])
    else:
        # 2-step look-ahead: Minimize d(current, u) + min(d(u, v) for v != u)
        # This is O(n^2) per call, which is fine for n=50.
        d_curr = distance_matrix[current_node, unvisited_nodes]
        
        # We need the minimum distance from each unvisited node to any other unvisited node.
        # sub_matrix is n_unvisited x n_unvisited
        sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        
        # Set diagonal to inf so we don't pick the node itself
        np.fill_diagonal(sub_matrix, np.inf)
        
        # min_next_dist[i] is the min distance from unvisited_nodes[i] to any other unvisited node
        min_next_dist = np.min(sub_matrix, axis=1)
        
        # Total score for each candidate
        scores = d_curr + min_next_dist
        
        best_idx = np.argmin(scores)
        return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
