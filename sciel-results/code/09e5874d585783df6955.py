"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix using a Savings Heuristic approach.
    
    This approach favors edges that result in significant distance savings compared
    to serving the nodes individually from the depot (Clarke-Wright savings).
    For depot-customer edges, it uses inverse distance to ensure connectivity.
    """
    n = distance_matrix.shape[0]
    D = distance_matrix
    
    # Calculate distances from the depot (node 0) to all other nodes
    d0 = D[0, :]
    
    # Calculate the savings matrix S_ij = d_0i + d_0j - d_ij
    # This represents the distance saved by connecting i and j directly
    # instead of going i->0->j
    # Note: S_0j and S_i0 are 0 by definition (d_00=0)
    savings = d0[:, np.newaxis] + d0[np.newaxis, :] - D
    
    # Initialize heuristic matrix
    heuristic = np.zeros_like(D)
    
    # For depot edges (row 0 and column 0), use inverse distance
    # This ensures ants can start and end their tours
    # We use inverse distance to prefer closer nodes when starting/ending
    inv_dist = 1.0 / (D + 1e-9)
    
    heuristic[0, :] = inv_dist[0, :]
    heuristic[:, 0] = inv_dist[:, 0]
    
    # For customer-customer edges (i > 0, j > 0), use the savings value
    # We add a small epsilon to ensure positivity
    # Savings are non-negative for Euclidean distances (triangle inequality)
    # We might want to scale them to be comparable to inverse distance values
    # Inverse distance ranges roughly from 1/sqrt(2) to 1/0.1 (if min dist is 0.1)
    # Savings range from 0 to ~2*sqrt(2)
    
    # Let's normalize savings to have a similar scale as inverse distance
    # Find max savings to scale
    max_savings = np.max(savings[1:, 1:])
    if max_savings > 0:
        # Scale savings to [0, 1] range, then boost
        # Or just use raw savings if they are comparable
        # Let's try: heuristic = savings + 1e-9
        # But inverse distance can be > 1.
        # Let's use a weighted combination or just raw savings if they drive the behavior.
        
        # A common trick: use 1 / (1 - savings/max_savings) ? No.
        # Let's just use the savings directly. The ACO pheromone will adapt.
        # However, if savings are much larger than inverse distance, depot edges might be ignored?
        # No, depot edges are forced by the algorithm logic (start/end).
        # The choice between customers is driven by customer-customer edges.
        
        # To be safe, let's normalize savings to be comparable to inverse distance.
        # Average inverse distance for customer-customer edges:
        avg_inv_dist = np.mean(inv_dist[1:, 1:])
        
        # Scale savings so their mean is comparable to avg_inv_dist
        if np.mean(savings[1:, 1:]) > 0:
            scale_factor = avg_inv_dist / np.mean(savings[1:, 1:])
            heuristic[1:, 1:] = savings[1:, 1:] * scale_factor
        else:
            heuristic[1:, 1:] = inv_dist[1:, 1:]
    else:
        heuristic[1:, 1:] = inv_dist[1:, 1:]
        
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
