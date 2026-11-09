"""Lagrangian Relaxation-based edge weight heuristic for CVRP ant colony."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix using a Lagrangian Relaxation approach.
    
    This implementation finds a Lagrange multiplier lambda that penalizes node demand
    in the edge costs. The optimal lambda is found by minimizing the total distance
    of a nearest-neighbor tour constructed on the modified cost matrix.
    The heuristic is derived from the inverse of the modified costs, sharpened
    to highlight edges that are locally optimal under the relaxed constraint.
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=float)
    demands[0] = 0.0  # Depot has no demand
    
    # Helper to compute total distance of a NN tour for a given lambda
    def tour_cost(lam):
        # Modified cost: W_ij = D_ij + lam * d_j
        # We construct a NN tour starting from depot.
        # Note: In CVRP, the "tour" is a sequence of customers.
        # We simulate the ant's path selection greedily based on W.
        
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        current = 0
        total_dist = 0.0
        
        for _ in range(n - 1):
            # Get costs from current to all unvisited
            costs = distance_matrix[current].copy()
            costs += lam * demands
            costs[visited] = np.inf
            
            # Find next node
            next_node = np.argmin(costs)
            if costs[next_node] == np.inf:
                break # Should not happen
            
            total_dist += costs[next_node]
            visited[next_node] = True
            current = next_node
            
        # Return to depot
        total_dist += distance_matrix[current, 0]
        
        return total_dist

    # Determine search range for lambda
    # Lambda represents cost per unit demand.
    # Max distance is roughly 1.4 (diagonal of unit square).
    # Demands are 1-9.
    # If lambda is 0, we ignore demand.
    # If lambda is large, we prioritize low demand nodes.
    # A reasonable range is [0, max_dist / min_demand].
    max_dist = np.max(distance_matrix)
    min_demand = np.min(demands[1:])
    max_lam = max_dist / min_demand * 2.0
    
    # Initial coarse search
    num_coarse = 50
    lams_coarse = np.linspace(0, max_lam, num_coarse)
    best_lam = lams_coarse[0]
    best_cost = tour_cost(best_lam)
    
    for lam in lams_coarse:
        c = tour_cost(lam)
        if c < best_cost:
            best_cost = c
            best_lam = lam
            
    # Refine search around best_lam
    # Determine step size from coarse search
    step = max_lam / num_coarse
    # Local refinement with 20 points in [best_lam - step, best_lam + step]
    lams_fine = np.linspace(best_lam - step, best_lam + step, 20)
    # Ensure non-negative lambda
    lams_fine = lams_fine[lams_fine >= 0]
    
    if len(lams_fine) > 0:
        for lam in lams_fine:
            c = tour_cost(lam)
            if c < best_cost:
                best_cost = c
                best_lam = lam

    # Construct the heuristic matrix
    # H_ij = 1 / (D_ij + lambda * d_j)^p
    # Using p=1.5 to sharpen the preference for attractive edges
    W = distance_matrix + best_lam * demands
    W = np.maximum(W, 1e-9)
    
    # Apply power to sharpen
    p = 1.5
    H = 1.0 / (W ** p)
    
    # Ensure finite values
    H = np.where(np.isfinite(H), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
