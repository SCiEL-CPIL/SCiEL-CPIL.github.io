import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Args:
        mu: Array of predicted means (shape: 2000,)
        sigma: Array of predicted standard deviations (shape: 2000,)
        f_best: Scalar, the best function value observed so far.
        cost: Array of costs for each candidate (shape: 2000,)
        
    Returns:
        scores: Array of acquisition scores (shape: 2000,). Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Calculate the mean uncertainty to set the exploration scale
    # Using mean(sigma) is a robust way to scale beta dynamically
    mean_sigma = np.mean(sigma)
    
    # Beta controls the trade-off between exploitation and exploration.
    # A value of 2.0 is a common heuristic for LCB in multimodal landscapes.
    # It is scaled by the current average uncertainty to adapt to the GP's state.
    beta = 2.0 * mean_sigma
    
    # Calculate the optimistic gap:
    # (f_best - mu) represents the potential improvement over the current best.
    # beta * sigma represents the uncertainty bonus.
    raw_score = (f_best - mu) + beta * sigma
    
    # Normalize by cost to prefer cheaper evaluations with similar expected gain.
    # Add a small epsilon to avoid division by zero.
    safe_cost = np.maximum(cost, 1e-9)
    
    scores = raw_score / safe_cost
    
    # Ensure all scores are finite to satisfy the contract
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
