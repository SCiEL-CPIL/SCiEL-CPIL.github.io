"""Cost-aware acquisition function for Bayesian optimization (powell 4d)

Structural Change: Replace the power-law cost penalty with a linear cost division 
(Expected Improvement per Unit Cost). This formulation directly optimizes the 
expected reduction in the best observed value per unit of budget spent. 
This encourages the algorithm to select points that offer the highest "return on 
investment" in terms of budget. A cost-adjusted exploration term is added to 
ensure sufficient exploration of uncertain regions, also weighted by cost efficiency.
"""
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per Unit Cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Lowest observed function value
    cost : array of shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x_scaled_to_01)
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Sanitize inputs to prevent numerical issues
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    # Ensure sigma is strictly positive for division
    sigma = np.maximum(sigma, 1e-8)
    # Ensure cost is strictly positive for division
    cost = np.maximum(cost, 1e-8)

    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Use stable calculation for Phi and phi
    # For large negative z, Phi(z) is ~0, EI is ~0
    # For large positive z, Phi(z) is ~1, phi(z) is ~0, EI is ~ f_best - mu
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)

    # Exploration term: UCB-style bonus
    # We add a term proportional to sigma to encourage exploration.
    # This is effectively adding a "virtual improvement" from uncertainty.
    beta = 1.0  # Exploration weight
    exploration = beta * sigma
    
    # Total expected benefit (in objective units)
    benefit = ei + exploration

    # Cost-aware scoring: Expected Benefit per Unit Cost
    # This linear division strongly penalizes expensive points,
    # prioritizing cheap points that offer reasonable improvement or exploration.
    scores = benefit / cost

    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
