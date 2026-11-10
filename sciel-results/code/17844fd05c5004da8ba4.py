"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an edge-attractiveness matrix for Ant Colony Optimization on the
    Orienteering Problem.
    
    The heuristic eta[i, j] is designed to favor moves to nodes with high prizes
    that are close to the current node and allow a feasible return to the origin
    within max_len.
    
    Formula: eta[i, j] = prize[j]^alpha / (distance[i, j]^beta * (1 + distance[j, 0]/max_len)^gamma)
    
    This balances prize collection, travel cost, and feasibility of returning home.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    
    # Distance from each node j to the origin (node 0)
    # dist_to_origin[j] = distance from node j back to node 0
    dist_to_origin = distance_matrix[:, 0].copy()
    
    # Prize at destination node j (broadcast to rows)
    prize_dest = prizes[None, :]  # shape (1, n)
    
    # Avoid division by zero: add small epsilon
    eps = 1e-9
    
    # Distance from i to j
    d_ij = distance_matrix + eps  # shape (n, n)
    
    # Feasibility factor: penalize nodes j where returning to origin is costly
    # If d[i,j] + d[j,0] is close to max_len, the move is risky
    # Use a smooth penalty that grows as d[j,0]/max_len approaches 1
    return_cost_ratio = dist_to_origin[None, :] / max_len  # shape (1, n)
    
    # Smooth feasibility penalty: 1 + return_cost_ratio^2
    # This gently penalizes nodes far from origin
    feasibility_penalty = 1.0 + return_cost_ratio ** 2  # shape (1, n)
    
    # Combined heuristic:
    # - High prize at destination
    # - Short distance from i to j
    # - Feasibility of returning to origin
    alpha = 2.0  # prize exponent
    beta = 1.5   # distance exponent
    gamma = 1.0  # feasibility exponent
    
    # eta[i, j] = prize[j]^alpha / (d[i,j]^beta * feasibility_penalty^gamma)
    numerator = np.maximum(prize_dest, eps) ** alpha  # shape (1, n)
    denominator = (d_ij ** beta) * (feasibility_penalty ** gamma)  # shape (n, n)
    
    eta = numerator / denominator  # shape (n, n)
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
