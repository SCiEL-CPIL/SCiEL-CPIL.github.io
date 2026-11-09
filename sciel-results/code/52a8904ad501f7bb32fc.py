"""Cost-aware acquisition for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB exploration with cost-aware exploitation.
    
    Uses a hybrid strategy:
    - UCB term for global exploration (decoupled from f_best)
    - Cost-normalized expected improvement for exploitation
    - Adaptive weighting based on the relative magnitude of uncertainty
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : ndarray, shape (2000,)
        Evaluation costs for each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero
    cost = np.maximum(cost, 1e-9)
    
    # Standardize mu and sigma for stability
    mu_mean = np.mean(mu)
    mu_std = np.std(mu) + 1e-9
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + 1e-9
    
    # UCB term: encourages exploration of uncertain regions
    # Beta controls exploration-exploitation balance
    beta = 2.0
    ucb = mu + beta * sigma
    
    # Cost-normalized UCB: prefer cheaper points with high uncertainty
    cost_normalized_ucb = -ucb / cost
    
    # Expected improvement term for exploitation
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI
    cost_normalized_ei = ei / cost
    
    # Combine: weight UCB more heavily in early stages (high uncertainty),
    # shift to EI as uncertainty decreases
    # Use the relative uncertainty to adaptively weight
    uncertainty_ratio = sigma / (sigma_mean + 1e-9)
    
    # Adaptive weight: higher uncertainty -> more UCB, lower -> more EI
    # Clip to [0.3, 0.7] to maintain balance
    w_ucb = np.clip(0.5 + 0.2 * (uncertainty_ratio - 1.0), 0.3, 0.7)
    w_ei = 1.0 - w_ucb
    
    # Combine the two terms
    scores = w_ucb * cost_normalized_ucb + w_ei * cost_normalized_ei
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
