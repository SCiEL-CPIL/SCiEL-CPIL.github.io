import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu: array of shape (N,)
    sigma: array of shape (N,)
    f_best: float, best observed value so far
    cost: array of shape (N,), cost of evaluating each candidate
    
    Returns:
    scores: array of shape (N,), higher is better
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate UCB (Lower Confidence Bound for minimization)
    # We want to minimize mu - beta * sigma
    # To maximize a score, we use -mu + beta * sigma
    beta = 1.5  # Exploration constant
    ucb_term = -mu + beta * sigma
    
    # Blend EI and UCB
    # EI is in objective units, UCB is in objective units.
    # We divide by cost to make it cost-aware.
    # We use a weighted sum. EI is generally more reliable for exploitation.
    # UCB is good for exploration.
    # Let's try a 50/50 blend, or maybe weight EI higher.
    # The inspiration said "Adaptive UCB-EI blend".
    # Let's try a simple blend first.
    
    # Normalize EI and UCB? No, they are in the same units (objective value).
    # However, EI can be 0 if mu > f_best. UCB can be negative.
    
    # Cost-aware score
    # We want to maximize (Improvement) / Cost
    # Let's combine them:
    # Score = (w_ei * EI + w_ucb * UCB_term) / Cost
    
    # To avoid division by zero or small costs (though cost >= 1), we are safe.
    # But we should handle the case where cost is very small? Cost is >= 1.
    
    # Let's try a dynamic blend. Early on, explore more.
    # We don't know the step count, but we can look at the range of mu.
    # If mu is very spread out, we are early.
    # Simpler: Just use a fixed blend.
    w_ei = 0.5
    w_ucb = 0.5
    
    # Combine
    combined = w_ei * ei + w_ucb * ucb_term
    
    # Cost adjustment
    # Dividing by cost favors cheaper points.
    scores = combined / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
