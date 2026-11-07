"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    
    Args:
        distance_matrix: (n, n) symmetric distance matrix
        coordinates: (n, 2) coordinates
        demands: (n,) demands (node 0 is depot with demand 0)
        capacity: vehicle capacity
        
    Returns:
        (n, n) heuristic matrix where larger = more attractive
    """
    n = distance_matrix.shape[0]
    d = distance_matrix.copy()
    
    # Ensure diagonal is not zero (avoid self-loops issues)
    np.fill_diagonal(d, np.inf)
    
    # Base attractiveness: inverse distance with power
    alpha = 2.0
    base = 1.0 / (d ** alpha)
    
    # Demand-based adjustment: penalize edges to high-demand customers
    # When going TO a customer j, the demand[j] consumes capacity
    # For depot->customer: mild penalty (starting fresh)
    # For customer->customer: stronger penalty (remaining capacity is uncertain)
    
    demand_ratio = demands / capacity  # (n,)
    
    # Create heuristic matrix
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                H[i, j] = 1e-9
                continue
            if i == 0:
                # Depot to customer: mild demand penalty
                # High demand customers are fine from depot (fresh load)
                penalty = 1.0 - 0.1 * demand_ratio[j]
            else:
                # Customer to customer: stronger penalty for high-demand destinations
                # Since remaining capacity is unknown statically, use moderate penalty
                penalty = 1.0 - 0.3 * demand_ratio[j]
            
            H[i, j] = base[i, j] * max(penalty, 0.1)
    
    # Ensure all entries are positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
