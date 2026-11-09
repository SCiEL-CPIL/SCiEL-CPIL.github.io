import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    score = mu - beta * sigma
    
    Args:
        mu: array of shape (N,) with GP mean predictions.
        sigma: array of shape (N,) with GP standard deviations.
        f_best: float, the best observed value so far.
        
    Returns:
        array of shape (N,) with acquisition scores.
    """
    # Ensure sigma is non-negative and finite
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Clip sigma to a small positive value to avoid numerical issues, though UCB is robust
    sigma = np.clip(sigma, 1e-10, 1e6)
    
    # Exploration parameter beta
    # A fixed beta of 1.0 is standard for UCB-1.
    # Given the lack of state (iteration count), we use a fixed value.
    beta = 1.0
    
    # Calculate UCB score for minimization
    # We want to minimize, so we prefer low mu.
    # We want to explore, so we prefer high sigma.
    # Thus, score = mu - beta * sigma
    scores = mu - beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
