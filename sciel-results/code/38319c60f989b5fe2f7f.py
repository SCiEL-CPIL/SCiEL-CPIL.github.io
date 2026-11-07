"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    # Heuristic for CVRP: combine inverse distance with demand-aware penalty.
    # For each customer j, compute a "cost" that includes the distance from i to j
    # plus a small penalty proportional to demand[j]. This encourages ants to
    # prefer shorter moves to low-demand customers, which helps pack vehicles
    # efficiently in the capacitated setting.
    #
    # The exponent is tuned to 1.5 to balance greediness and exploration in the
    # 30-iteration ACO run.
    eps = 1e-9
    
    # Ensure distances are strictly positive
    d = np.maximum(distance_matrix, eps)
    
    # Demand-aware cost: distance + small demand penalty
    # demands is an array of length n (including depot at index 0 with demand 0)
    # We want to penalize visiting high-demand customers more heavily
    demand_penalty = 0.1 * demands[None, :]  # shape (1, n)
    
    # Total cost matrix: shape (n, n)
    cost = d + demand_penalty
    
    # Heuristic: inverse of cost raised to power 1.5
    eta = 1.0 / (cost ** 1.5)
    
    # Handle any potential non-finite values
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
