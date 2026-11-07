"""Heuristic matrix for ant colony optimization of orienteering."""
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness heuristic matrix for the ant colony.
    
    Design: Potential-based Heuristic with Squared Potential.
    1. Calculate Node Potential: prize[j] - lambda * d[j,0]
       This accounts for the prize gain and the cost of returning to start.
    2. Square the Potential: (prize[j] - lambda * d[j,0])^2
       This amplifies the attractiveness of high-value, low-cost nodes,
       creating a steeper heuristic landscape for faster ACO convergence.
    3. Divide by Step Cost: / (d[i,j] + eps)
       This penalizes long steps from the current node i.
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # 1. Calculate Node Potential
    # lambda_return controls the penalty for distance from start.
    # With max_len=3 and unit square distances, 0.5-0.7 is a reasonable range.
    # We keep 0.5 as it was in the parent, but the squaring will change the dynamics.
    lambda_return = 0.5
    d_j0 = distance_matrix[:, 0] 
    node_potential = prizes - lambda_return * d_j0
    
    # Floor to a small positive number to ensure all nodes have some base attractiveness
    # and to avoid zero division or negative values in the squaring step.
    node_potential = np.maximum(node_potential, 1e-6)
    
    # 2. Square the Potential
    # This increases the contrast between "good" and "bad" nodes.
    # Nodes with slightly higher net value become significantly more attractive.
    squared_potential = node_potential ** 2
    
    # 3. Calculate Step Attractiveness
    d_ij = distance_matrix
    eps = 1e-6
    step_cost = d_ij + eps
    
    # Heuristic: Squared Potential of destination / Step Cost
    heuristic = squared_potential[None, :] / step_cost
    
    # Ensure finite and positive
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Zero out the diagonal to prevent self-loops
    np.fill_diagonal(heuristic, 0.0)
    
    return heuristic
