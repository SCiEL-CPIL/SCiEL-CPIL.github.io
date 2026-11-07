import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB).
    
    For minimization, we aim to find points with low predicted values and high uncertainty.
    The Lower Confidence Bound is defined as:
    LCB(x) = mu(x) - kappa * sigma(x)
    
    We select the point with the lowest LCB, which is equivalent to maximizing:
    Score(x) = -mu(x) + kappa * sigma(x)
    
    To account for evaluation costs, we divide by the cost:
    Final Score(x) = (-mu(x) + kappa * sigma(x)) / cost(x)
    
    Higher score indicates a more promising point to evaluate next.
    """
    # Ensure sigma is non-negative to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Exploration-exploitation trade-off parameter
    kappa = 1.5
    
    # Calculate the negative Lower Confidence Bound
    # Maximizing this is equivalent to minimizing the LCB
    lcb_value = -mu + kappa * sigma
    
    # Cost-aware score
    scores = lcb_value / cost
    
    # Handle any potential non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
