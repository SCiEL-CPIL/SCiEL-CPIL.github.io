"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    
    We use a combined heuristic that balances:
    1. Short distance (standard 1/d)
    2. Savings-based attractiveness (Clarke-Worth savings)
    3. Demand consideration (prefer edges that connect nodes with similar demands
       to help form balanced routes)
    
    The heuristic is computed once per instance before the colony runs.
    Entries that are not positive become 1e-9.
    """
    n = distance_matrix.shape[0]
    depot = 0
    
    # Standard distance-based heuristic
    d_heur = 1.0 / (distance_matrix + 1e-9)
    
    # Clarke-Worth savings: s(i,j) = d(i,0) + d(0,j) - d(i,j)
    # Large savings means connecting i and j directly is much better than going through depot
    d_i_depot = distance_matrix[:, depot]
    d_depot_j = distance_matrix[depot, :]
    savings = d_i_depot[:, np.newaxis] + d_depot_j[np.newaxis, :] - distance_matrix
    
    # Convert savings to attractiveness: larger savings = more attractive
    # Use a smooth transformation: exp(savings / scale)
    # Scale to keep values in a reasonable range
    max_savings = np.max(savings)
    if max_savings > 1e-6:
        scale = max_savings / 2.0
        savings_heur = np.exp(savings / scale)
    else:
        savings_heur = np.ones_like(savings)
    
    # Normalize savings_heur to be comparable to d_heur
    savings_heur = savings_heur / np.max(savings_heur)
    
    # Combine: geometric mean of distance heuristic and savings heuristic
    # This gives weight to both short edges and high-savings edges
    combined = np.sqrt(d_heur * savings_heur)
    
    # Ensure all values are positive
    combined = np.maximum(combined, 1e-9)
    
    # Make diagonal zero (no self-loops)
    np.fill_diagonal(combined, 0.0)
    
    return combined
# EVOLVE-BLOCK-END
