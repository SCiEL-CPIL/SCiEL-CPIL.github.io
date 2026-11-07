import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and UCB.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP std dev predictions  
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,) - higher means evaluate next
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid division issues
    sigma = np.clip(sigma, 1e-8, None)
    
    # 1. Expected Improvement (EI)
    z = (f_best - mu) / sigma
    # Handle case where sigma is very small
    # If sigma is 0, z is inf. 
    # norm.cdf(inf) = 1, norm.pdf(inf) = 0.
    # EI = (f_best - mu) * 1 + 0 = f_best - mu.
    # This is correct.
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # 2. UCB Component (for minimization)
    # We want to maximize the score.
    # Low mu is good. High sigma is good (uncertainty).
    # Score_ucb = -mu + beta * sigma
    beta = 2.5
    ucb_score = -mu + beta * sigma
    
    # 3. Combine
    # Both EI and ucb_score are in objective units.
    # They naturally balance: EI is 0 when no improvement expected, 
    # ucb_score drives exploration when sigma is high.
    combined = ei + ucb_score
    
    # 4. Cost Awareness
    # Divide by cost to favor cheaper points
    scores = combined / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
