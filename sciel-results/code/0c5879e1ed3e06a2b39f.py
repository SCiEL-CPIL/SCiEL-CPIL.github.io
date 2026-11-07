"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    
    Strategy:
    1. Use a base heuristic based on inverse distance with a demand-aware penalty.
    2. Perform a lightweight parameter search using a greedy nearest-neighbor
       construction with demand constraints to find optimal (alpha, beta) parameters.
    3. The heuristic is static and computed once per instance.
    
    The heuristic entry eta[i, j] represents the attractiveness of moving from i to j.
    Larger values are more attractive.
    """
    n = distance_matrix.shape[0]
    d = distance_matrix.astype(float).copy()
    # Avoid division by zero on diagonal
    np.fill_diagonal(d, 1.0)
    
    # Time budget: aim to finish within 10 seconds to be safe
    start_time = time.time()
    deadline = start_time + 10.0
    
    # Parameter grid
    # alpha: exponent on distance (higher = more local)
    # beta: exponent on demand penalty (higher = more penalty for high demand)
    # gamma: weight of demand penalty vs distance
    
    alphas = [1.0, 1.5, 2.0, 2.5, 3.0]
    betas = [0.0, 0.5, 1.0, 1.5]
    
    best_score = float('inf')
    best_alpha = 2.0
    best_beta = 0.0
    
    # Pre-compute demand factors
    # We penalize edges to high-demand nodes if they are "far" in some sense?
    # Or just a static penalty for high demand to encourage them to be grouped?
    # Actually, for CVRP, a good heuristic often encourages visiting nearby nodes.
    # Demand constraint is handled by the ant's decision rule (feasibility check).
    # So the heuristic should primarily reflect distance, possibly adjusted for
    # "difficulty" of the node (high demand).
    
    # Let's try: eta[i,j] = (1 / d[i,j]^alpha) * (1 - beta * demands[j]/capacity)^gamma
    # Or simpler: eta[i,j] = 1 / (d[i,j]^alpha * (1 + beta * demands[j]/capacity))
    
    def greedy_cost(alpha, beta):
        # Construct heuristic matrix
        # eta[i,j] = 1 / (d[i,j]^alpha) * (1 / (1 + beta * demands[j]/capacity))
        # We want to maximize eta when choosing next node in greedy construction
        
        # Pre-compute column factors for demand penalty
        # penalty_j = 1 / (1 + beta * demands[j]/capacity)
        penalty = 1.0 / (1.0 + beta * (demands / capacity))
        
        # For each i, eta[i,j] = (1/d[i,j]^alpha) * penalty[j]
        # We can compute this on the fly or pre-compute the distance part.
        
        unvisited = list(range(1, n))
        current = 0
        remaining_capacity = capacity
        total_dist = 0.0
        
        while unvisited:
            valid_next = []
            best_eta = -1
            best_node = None
            
            for j in unvisited:
                if demands[j] <= remaining_capacity:
                    # eta = (1 / d[i,j]^alpha) * penalty[j]
                    eta_val = (1.0 / (d[current, j] ** alpha)) * penalty[j]
                    if eta_val > best_eta:
                        best_eta = eta_val
                        best_node = j
            
            if best_node is None:
                # Should not happen if capacity is large enough for at least one
                # but if all remaining demands are too high, return to depot
                total_dist += d[current, 0]
                current = 0
                remaining_capacity = capacity
                continue
            
            total_dist += d[current, best_node]
            remaining_capacity -= demands[best_node]
            current = best_node
            unvisited.remove(best_node)
        
        total_dist += d[current, 0]
        return total_dist
    
    # Evaluate grid
    for a in alphas:
        if time.time() > deadline - 5.0:
            break
        for b in betas:
            if time.time() > deadline - 5.0:
                break
            score = greedy_cost(a, b)
            if score < best_score:
                best_score = score
                best_alpha = a
                best_beta = b
    
    # Construct final heuristic matrix
    alpha = best_alpha
    beta = best_beta
    
    penalty = 1.0 / (1.0 + beta * (demands / capacity))
    
    # H[i,j] = (1 / d[i,j]^alpha) * penalty[j]
    # Vectorized computation
    # d^alpha
    d_alpha = d ** alpha
    inv_d_alpha = 1.0 / d_alpha
    # Multiply by penalty[j] (broadcast along rows)
    H = inv_d_alpha * penalty[None, :]
    
    # Set diagonal to 0 or small value (should not be used)
    np.fill_diagonal(H, 1e-9)
    
    # Clean up non-finite or non-positive values
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    # Normalize to max 1.0 (optional, but keeps numbers manageable)
    h_max = np.max(H)
    if h_max > 0:
        H = H / h_max
    
    return H
# EVOLVE-BLOCK-END
