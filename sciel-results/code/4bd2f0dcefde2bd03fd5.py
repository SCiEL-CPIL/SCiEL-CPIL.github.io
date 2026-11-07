"""Heuristic matrix for ant colony optimization of orienteering."""
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness heuristic matrix for the ant colony.
    
    Design: Efficiency-based Heuristic (Prize-to-Cost Ratio).
    This structural change moves from additive potential subtraction to a
    multiplicative efficiency model.
    
    1. Calculate Node Efficiency: prize[j] / (d[j,0] + eps)
       This measures the prize gained per unit of "return cost".
       Nodes close to the start with high prizes are considered highly efficient.
    2. Calculate Edge Attractiveness: Node Efficiency[j] / (d[i,j] + eps)
       This further penalizes the step distance from i to j.
       
    This formulation strongly favors nodes that are both high-value and
    geographically convenient (close to start and close to current node),
    creating a sharper heuristic landscape than linear potential subtraction.
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    eps = 1e-6
    
    # 1. Node Efficiency: Prize relative to return distance
    # Avoid division by zero by adding eps
    d_j0 = distance_matrix[:, 0]
    node_efficiency = prizes / (d_j0 + eps)
    
    # Ensure non-negative (prizes are non-negative, distances are non-negative)
    node_efficiency = np.maximum(node_efficiency, 0.0)
    
    # 2. Edge Attractiveness
    # Divide node efficiency by the step distance
    d_ij = distance_matrix + eps
    heuristic = node_efficiency[None, :] / d_ij
    
    # Ensure finite and positive
    # The contract states entries that are not positive become 1e-9.
    # We explicitly set non-positive/non-finite values to 1e-9 here to be safe.
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Zero out the diagonal to prevent self-loops
    np.fill_diagonal(heuristic, 0.0)
    
    # Re-apply positive floor for diagonal if needed? 
    # Usually diagonal is ignored or set to 0. 
    # If the ACO code checks `heuristic > 0`, 0 will be replaced by 1e-9.
    # This is fine.
    
    return heuristic
