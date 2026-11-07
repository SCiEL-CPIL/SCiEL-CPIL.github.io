"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Multiplicative Guided Local Search penalty.
    
    Instead of adding a linear penalty, this multiplies the edge distances
    by a factor that depends on the logarithm of the usage count.
    This preserves the relative scale of distances better and avoids
    degenerate cases where one edge becomes infinitely expensive.
    
    Edges in the current local optimum tour receive an additional
    multiplicative penalty to encourage escaping the current basin.
    """
    n = edge_distance.shape[0]
    
    # Logarithmic scaling of usage to prevent explosive growth of penalties
    # log1p(x) is ~x for small x, but grows slowly for large x
    usage_log = np.log1p(edge_n_used)
    
    # Base multiplicative factor: 1 + alpha * log(1 + count)
    # alpha = 0.2 means an edge used 10 times gets a factor of ~1.48
    # An edge used 20 times gets a factor of ~1.60
    alpha = 0.2
    factor = 1.0 + alpha * usage_log
    
    # Identify edges in the current local optimum tour
    tour_mask = np.zeros_like(edge_distance, dtype=bool)
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        tour_mask[i, j] = True
        tour_mask[j, i] = True
        tour_mask[i, k] = True
        tour_mask[k, i] = True
    
    # Apply an extra multiplicative penalty to edges in the current tour
    # beta = 0.5 means tour edges get an additional 1.5x multiplier
    # This strongly discourages keeping the same structure
    beta = 0.5
    factor[tour_mask] *= (1.0 + beta)
    
    # Apply the multiplicative penalty
    penalized = edge_distance * factor
    
    # Ensure symmetry (should already be symmetric, but good practice)
    penalized = (penalized + penalized.T) / 2.0
    
    # Ensure diagonal is zero
    np.fill_diagonal(penalized, 0.0)
    
    # Ensure all values are finite
    penalized = np.where(np.isfinite(penalized), penalized, edge_distance)
    
    return penalized
# EVOLVE-BLOCK-END
