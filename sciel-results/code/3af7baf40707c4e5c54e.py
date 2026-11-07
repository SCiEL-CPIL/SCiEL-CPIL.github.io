import numpy as np

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ant colony optimization of the
    multiple-constraint knapsack problem.
    
    Uses a shadow price heuristic where the cost of each resource is scaled
    by the ratio of total available weight in that dimension to the capacity.
    This penalizes items that consume resources in tight constraints more heavily.
    
    Args:
        prizes: array of shape (5, 100) with prize values in [0, 1]
        weights: array of shape (5, 100) with weight values in [0, 1]
        
    Returns:
        array of shape (100,) with one attractiveness score per item
    """
    # Total prize for each item (sum across the 5 dimensions)
    # Shape: (100,)
    P = prizes.sum(axis=0)
    
    # Total weight for each constraint (sum across the 100 items)
    # Shape: (5,)
    S = weights.sum(axis=1)
    
    # Fixed capacity for each constraint
    C = 25.0
    
    # Estimate shadow prices (lambda) for each constraint.
    # lambda_k = S_k / C_k
    # If S_k > C_k, the constraint is tight (binding), so lambda_k > 1.
    # This increases the "cost" of using that resource in the heuristic.
    # If S_k < C_k, the constraint is loose, so lambda_k < 1.
    # This decreases the "cost" of using that resource.
    lambda_k = S / C
    
    # Calculate effective weight for each item.
    # W_eff[j] = sum_k lambda_k * W[k, j]
    # weights is (5, 100), lambda_k is (5,)
    # We need to broadcast lambda_k to (5, 1) to multiply with weights.
    W_eff = np.sum(weights * lambda_k[:, np.newaxis], axis=0)
    
    # Heuristic score: Prize / Effective Weight
    # Items with high prize and low effective weight are more attractive.
    # Add a small epsilon to avoid division by zero.
    H = P / (W_eff + 1e-9)
    
    # Ensure all scores are positive (ACO requires positive heuristics)
    H = np.maximum(H, 1e-9)
    
    return H
