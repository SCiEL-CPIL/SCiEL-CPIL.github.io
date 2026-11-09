"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # Safety margin before 150s kill


def _time_remaining() -> float:
    """Return seconds remaining before the deadline."""
    return _DEADLINE - time.time()


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour using a lookahead-based heuristic.
    
    Args:
        current_node: int - the current city
        destination_node: int - the final destination (usually the start city)
        unvisited_nodes: np.ndarray - array of unvisited city indices
        distance_matrix: np.ndarray - n x n matrix of pairwise distances
        
    Returns:
        int - the index of the next city to visit
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not happen, but return a safe fallback
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If we're running out of time, fall back to nearest neighbor
    if _time_remaining() < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Distance from current to each unvisited node
    d_current = distance_matrix[current_node, unvisited_nodes]  # shape (n_unvisited,)
    
    # Distance from each unvisited node to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]  # shape (n_unvisited,)
    
    # For 2-step lookahead: pick node i that minimizes d_current[i] + min_j(d[i,j] + d_dest[j])
    # For efficiency, we approximate the second-step cost.
    
    if n_unvisited <= 20:
        # Full 2-step lookahead: for each candidate i, find best j != i
        # cost(i) = d_current[i] + min_{j != i} (d_matrix[i,j] + d_dest[j])
        best_cost = np.inf
        best_idx = 0
        
        # Get the submatrix of distances among unvisited nodes
        d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]  # (n_unvisited, n_unvisited)
        
        for i in range(n_unvisited):
            # For step 2, we can go to any other unvisited node j
            # cost of going i -> j -> dest is d_unvisited[i, j] + d_dest[j]
            # We want min over j != i
            costs_j = d_unvisited[i, :] + d_dest  # (n_unvisited,)
            # Exclude j == i
            costs_j[i] = np.inf
            min_second = np.min(costs_j)
            total = d_current[i] + min_second
            if total < best_cost:
                best_cost = total
                best_idx = i
        
        return int(unvisited_nodes[best_idx])
    
    else:
        # For larger sets, use a hybrid: combine immediate distance with a 
        # weighted destination distance (a form of "greedy with lookahead")
        # This is cheaper than full 2-step but better than pure NN.
        
        # Use a soft combination: score = d_current + alpha * d_dest
        # where alpha balances short-term vs long-term
        # A good alpha is around 0.3-0.5 for TSP
        
        alpha = 0.4
        scores = d_current + alpha * d_dest
        
        # But also consider: for the last few nodes, pure destination distance matters more
        # Adjust alpha based on how many nodes remain
        if n_unvisited <= 5:
            alpha = 0.8
        elif n_unvisited <= 10:
            alpha = 0.6
        
        scores = d_current + alpha * d_dest
        return int(unvisited_nodes[np.argmin(scores)])

# EVOLVE-BLOCK-END
