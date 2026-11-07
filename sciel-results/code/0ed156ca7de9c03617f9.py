import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided Local Search perturbation: Hard Cut Strategy.
    
    Instead of accumulating penalties on all previously used edges, this strategy
    applies a high penalty factor ONLY to the edges in the current local-optimal tour.
    This forces 2-opt to sever specific connections in the current tour to escape the
    local optimum, while leaving the relative distances of all other candidate edges
    (including those used in past rounds but not in the current tour) undistorted.
    
    This preserves the geometric structure of the search space more faithfully than
    cumulative penalties, which can distort the landscape by making many non-optimal
    edges artificially expensive over time.
    """
    start_time = time.time()
    n = edge_distance.shape[0]
    
    if n == 0:
        return edge_distance.copy()
        
    d_out = edge_distance.copy()
    
    # Identify edges in the current local optimal tour
    tour_edges = np.zeros((n, n), dtype=bool)
    
    try:
        for i in range(n):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + 1) % n]
            if 0 <= u < n and 0 <= v < n:
                tour_edges[u, v] = True
                tour_edges[v, u] = True
    except Exception:
        return d_out

    # Apply a high penalty factor only to tour edges.
    # We use a factor of 5.0. This is large enough to make 2-opt strongly prefer
    # swapping out any tour edge, but not so large that it breaks the matrix symmetry
    # or causes numerical issues. Since 2-opt considers swaps of two edges, making
    # all tour edges expensive encourages the algorithm to find a pair of swaps that
    # removes two tour edges and adds two non-tour edges, effectively "cutting" the
    # current tour in two places and reconnecting.
    
    # We do NOT apply penalties to edge_n_used here, as per the strategy directive.
    # The idea is that the "memory" of past failures is less important than the
    # immediate need to break the current local optimum's structure.
    
    penalty_factor = 5.0
    d_out[tour_edges] *= penalty_factor
    
    # Safety check: ensure finite values
    if not np.all(np.isfinite(d_out)):
        return edge_distance.copy()
        
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 2.5:
        return edge_distance.copy()
        
    return d_out
