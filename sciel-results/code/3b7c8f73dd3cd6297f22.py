import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Expected Improvement (EI) and UCB with cost normalization.
    
    Parameters:
    mu (np.ndarray): Mean predictions from the GP, shape (N,)
    sigma (np.ndarray): Standard deviation predictions from the GP, shape (N,)
    f_best (float): The best (lowest) observed objective value so far.
    cost (np.ndarray): Cost of evaluating each point, shape (N,)
    
    Returns:
    np.ndarray: Acquisition scores, higher is better, shape (N,)
    """
    # Numerical stability
    eps = 1e-9
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Exploration parameter xi for EI
    # xi > 0 encourages exploring regions with high uncertainty
    xi = 0.05
    
    # Calculate z-score for modified EI
    z = (f_best - mu - xi * sigma_safe) / sigma_safe
    
    # Calculate CDF and PDF of standard normal distribution
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement with xi
    # EI = (f_best - mu - xi*sigma) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu - xi * sigma_safe) * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)
    
    # UCB term: mean + kappa * sigma
    # UCB encourages exploration in uncertain regions
    kappa = 0.5
    ucb = mu + kappa * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    # Use the range of each component for normalization
    ei_range = np.max(ei) - np.min(ei)
    ucb_range = np.max(ucb) - np.min(ucb)
    
    ei_norm = (ei - np.min(ei)) / (ei_range + eps)
    ucb_norm = (ucb - np.min(ucb)) / (ucb_range + eps)
    
    # Combine EI and UCB with a weighted sum
    # Weight EI more heavily since we're minimizing
    w_ei = 0.7
    w_ucb = 0.3
    combined = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Cost-aware scoring
    # Divide by cost^gamma to penalize expensive evaluations
    # gamma < 1 makes the score less sensitive to cost
    gamma = 0.7
    scores = combined / np.power(cost_safe, gamma)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
