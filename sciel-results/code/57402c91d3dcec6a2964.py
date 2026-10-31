"""Rewritten program: TSP constructive heuristic with 3-step lookahead and adaptive destination bias."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 3-step lookahead with adaptive destination bias.
    
    The cost for candidate c is approximated as:
    d(current, c) + min_{b != c} ( d(c, b) + min_{a != b, c} ( d(b, a) + gamma * d(a, dest) ) )
    
    This is computed in O(N^2) time per step using vectorized operations.
    gamma adapts to the number of remaining nodes, increasing the pull towards 
    the destination as the tour nears completion.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline early
    if _is_deadline_passed():
        # Fallback to pure nearest neighbor
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    # Compute distances from current node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from unvisited nodes to destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Adaptive gamma: increases as n_unvisited decreases
    # Total nodes is assumed to be 50 based on task description, but we can estimate N_total
    # from the size of distance_matrix if needed, or just use n_unvisited relative to a max.
    # Let's assume max nodes is 50.
    n_total = 50 
    if n_total < n_unvisited:
        n_total = n_unvisited
        
    # gamma goes from 0.1 (early) to 1.0 (late)
    # Linear interpolation based on fraction of nodes remaining
    frac_remaining = n_unvisited / n_total
    gamma = 0.1 + 0.9 * (1.0 - frac_remaining)
    
    # Get the submatrix of distances among unvisited nodes
    submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    if n_unvisited > 2:
        # Step 1: Compute M[j] = min_{k != j} ( submatrix[j, k] + gamma * dists_to_dest[k] )
        # We create a matrix V where V[j, k] = submatrix[j, k] + gamma * dists_to_dest[k]
        V = submatrix + gamma * dists_to_dest[np.newaxis, :]
        
        # We need the min of each row V[j, :], excluding the diagonal element V[j, j]
        # Set diagonal to inf
        V_diag_inf = V.copy()
        np.fill_diagonal(V_diag_inf, np.inf)
        
        # M[j] is the min of row j in V_diag_inf
        M = np.min(V_diag_inf, axis=1)
        
        # Step 2: Compute Cost[i] = min_{j != i} ( submatrix[i, j] + M[j] )
        # Create matrix W where W[i, j] = submatrix[i, j] + M[j]
        W = submatrix + M[np.newaxis, :]
        
        # We need the min of each row W[i, :], excluding the diagonal element W[i, i]
        W_diag_inf = W.copy()
        np.fill_diagonal(W_diag_inf, np.inf)
        
        # Cost[i] is the min of row i in W_diag_inf
        lookahead_costs = np.min(W_diag_inf, axis=1)
        
        # Total cost for candidate i: d(current, c_i) + lookahead_costs[i]
        total_costs = dists_from_current + lookahead_costs
        
        best_idx = np.argmin(total_costs)
    else:
        # Only 2 nodes left, just pick the one closer to current
        best_idx = np.argmin(dists_from_current)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
