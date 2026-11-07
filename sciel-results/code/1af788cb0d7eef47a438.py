"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty with a repulsive force field.
    
    This implementation models the penalty as a continuous field that is highest
    at the edges of the current local optimum tour and decays with the metric
    distance to the nearest tour edge. This encourages 2-opt to break the current
    tour structure and explore distant regions of the solution space.
    """
    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()
    
    D = edge_distance
    
    # Identify tour edges
    # We compute the "closeness" of every pair (i,j) to the set of tour edges.
    # Closeness is defined as the minimum over all tour edges (a,b) of
    # min(D[i,a] + D[j,b], D[i,b] + D[j,a]).
    
    # Initialize min_closeness
    min_closeness = np.full((n, n), np.inf)
    
    # Iterate over tour edges to compute the minimum closeness field
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        
        # Vectors of distances from all nodes to a and b
        da = D[:, a]
        db = D[:, b]
        
        # Compute closeness contribution from this tour edge
        # c1[i,j] = D[i,a] + D[j,b]
        # c2[i,j] = D[i,b] + D[j,a]
        c1 = da[:, None] + db[None, :]
        c2 = db[:, None] + da[None, :]
        
        # Take the minimum of the two orientations
        c = np.minimum(c1, c2)
        
        # Update global minimum
        min_closeness = np.minimum(min_closeness, c)
    
    # Normalize usage counts
    max_used = edge_n_used.max()
    if max_used > 0:
        norm_used = edge_n_used / max_used
    else:
        norm_used = np.zeros_like(edge_n_used)
    
    # Parameters for the force field
    # beta controls the width of the exclusion zone
    # gamma controls the strength of the penalty
    beta = 8.0
    gamma = 1.0
    
    # Calculate the force field intensity
    # High intensity (close to 1) for edges near the tour, low (close to 0) for others
    force_field = np.exp(-beta * min_closeness)
    
    # Construct the penalty factor
    # Base penalty is 1.0 (no change).
    # We add a penalty proportional to the force field and the usage history.
    # The term (1.0 + 0.5 * norm_used) ensures that even unused edges near the tour
    # are penalized, but used edges are penalized more heavily.
    penalty_factor = 1.0 + gamma * force_field * (1.0 + 0.5 * norm_used)
    
    # Apply penalty to the distance matrix
    result = D * penalty_factor
    
    # Enforce symmetry and zero diagonal
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 0.0)
    
    # Ensure numerical stability
    result = np.nan_to_num(result, nan=0.0, posinf=D.max(), neginf=0.0)
    
    return result
# EVOLVE-BLOCK-END
