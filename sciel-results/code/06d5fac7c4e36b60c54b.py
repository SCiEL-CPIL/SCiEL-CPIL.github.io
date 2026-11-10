# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Heuristic matrix for ant colony optimization of CVRP.
    
    We use a composite heuristic that combines:
    1. Inverse distance (closer nodes are more attractive)
    2. Demand-to-capacity ratio (lower ratio means more room, slightly less attractive 
       for filling up routes, but we want to balance)
    3. A penalty for nodes that are far from the depot relative to their demand
    
    The key insight: in CVRP, we want to create tight routes. So we want to prefer
    nodes that are close to each other (clustered) and that allow us to fill capacity.
    
    A good heuristic: prefer nodes that are close to the current position AND have
    moderate demand (not too small, not too large). We use:
    eta[i,j] = (1 / (d[i,j] + eps)) * (1 + 0.1 * demands[j] / capacity)
    
    But actually, let's think differently. The heuristic should be static and computed
    once. Let's use a combination that encourages:
    - Short edges
    - Balanced loads (prefer nodes with demand closer to average)
    
    Let's use: eta[i,j] = 1 / (d[i,j] + eps) * (1 / (1 + |demands[j] - avg_demand| / avg_demand))
    
    Actually, simpler and more effective for CVRP: use inverse distance weighted by
    how "central" the node is (closer to centroid is better for routing).
    
    Let me use a well-known approach: eta[i,j] = 1 / (d[i,j] + eps)
    but with a twist: we can make it asymmetric by considering the return-to-depot cost.
    
    For a static heuristic, let's use:
    eta[i,j] = 1 / (d[i,j] + 0.5 * d[j,0] + eps)
    
    This prefers nodes that are close to i AND close to the depot (so the return trip is short).
    """
    n = distance_matrix.shape[0]
    eps = 1e-9
    
    # Distance to depot for each node
    dist_to_depot = distance_matrix[:, 0].copy()
    
    # Composite heuristic: prefer short edges and nodes that are close to depot
    # eta[i,j] = 1 / (d[i,j] + alpha * d[j,0])
    alpha = 0.3
    
    # Build the heuristic matrix
    heuristic = np.zeros_like(distance_matrix, dtype=np.float64)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                heuristic[i, j] = 0.0
            else:
                heuristic[i, j] = 1.0 / (distance_matrix[i, j] + alpha * dist_to_depot[j] + eps)
    
    # Ensure positive values
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
