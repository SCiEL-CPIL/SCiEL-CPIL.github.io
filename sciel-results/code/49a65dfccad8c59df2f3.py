"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Budget-aware expected utility heuristic for ACO orienteering.
    
    eta_ij = (pi_j / (d_ij + d_j0)^1.5) * exp(-lambda * (d_ij + d_j0) / L_max)
    
    This penalizes moves that consume a disproportionate fraction of the
    remaining length budget, preventing ants from getting stuck far from
    the depot with insufficient range to return.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    max_len = float(max_len)
    
    n = len(prizes)
    
    # Total cost of moving i -> j -> 0
    # d_ij is distance_matrix[i, j], d_j0 is distance_matrix[j, 0]
    d_j0 = distance_matrix[:, 0]  # shape (n,)
    total_cost = distance_matrix + d_j0[None, :]  # shape (n, n)
    
    # Ensure minimum cost to avoid division by zero
    eps = 1e-9
    total_cost_safe = np.maximum(total_cost, eps)
    
    # Prize of destination node j
    pi_j = prizes[None, :]  # shape (1, n)
    
    # Core: pi_j / (d_ij + d_j0)^1.5
    core = pi_j / (total_cost_safe ** 1.5)
    
    # Budget penalty: exp(-lambda * (d_ij + d_j0) / L_max)
    # lambda = 1.0 balances exploration vs. budget conservation
    lam = 1.0
    budget_penalty = np.exp(-lam * total_cost_safe / max_len)
    
    eta = core * budget_penalty
    
    # Set infeasible transitions (where total_cost > max_len) to very small value
    # Ants can only move if length_so_far + d[i,j] + d[j,0] <= max_len
    # Since length_so_far >= 0, if d[i,j] + d[j,0] > max_len, the move is infeasible
    infeasible = total_cost > max_len
    eta[infeasible] = eps
    
    # Ensure all values are positive and finite
    eta = np.maximum(eta, eps)
    eta = np.where(np.isfinite(eta), eta, eps)
    
    return eta
# EVOLVE-BLOCK-END
