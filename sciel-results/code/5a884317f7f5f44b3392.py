import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a blend of Expected Improvement (EI) and Upper Confidence Bound (UCB),
    normalized by the square root of the evaluation cost.
    
    Parameters:
    mu (np.ndarray): Mean predictions from the GP, shape (N,)
    sigma (np.ndarray): Standard deviation predictions from the GP, shape (N,)
    f_best (float): The best (lowest) observed objective value so far.
    cost (np.ndarray): Cost of evaluating each point, shape (N,)
    
    Returns:
    np.ndarray: Acquisition scores, higher is better, shape (N,)
    """
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: return simple EI-based scores to finish quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0) / cost_safe

    # Numerical stability
    eps = 1e-9
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Exploration parameter xi for modified EI
    xi = 0.05
    
    # Calculate z-score for modified EI
    z = (f_best - mu - xi * sigma_safe) / sigma_safe
    
    # Calculate CDF and PDF of standard normal distribution
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement with xi
    ei = (f_best - mu - xi * sigma_safe) * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)
    
    # UCB component: encourages exploration of uncertain regions
    # beta controls the trade-off between exploitation and exploration
    beta = 1.5
    ucb = mu + beta * sigma_safe
    
    # For minimization, we want to maximize the potential improvement
    # EI is already a measure of expected improvement (higher is better)
    # UCB gives a lower bound on the possible value (we want low values)
    # So for minimization, we should use -UCB or (f_best - UCB)
    # Let's use a combination: EI + weight * (f_best - mu - beta*sigma)
    # Actually, for minimization, UCB should be: mu - beta*sigma (lower bound)
    # We want to maximize the chance of finding a lower value
    # So the acquisition should be high when we expect low values
    # EI already captures this. Let's blend EI with a UCB-like term.
    
    # UCB for minimization: we want to explore where the lower confidence bound is low
    # Lower confidence bound: mu - beta * sigma
    # We want to maximize (f_best - lower_bound) which is (f_best - mu + beta*sigma)
    # But this doesn't account for the actual improvement.
    
    # A better approach: use EI as the primary term and add a small UCB term
    # to encourage exploration of high-uncertainty regions
    ucb_term = beta * sigma_safe
    
    # Blend EI and UCB term
    # w_ei = 0.8, w_ucb = 0.2
    w_ei = 0.8
    w_ucb = 0.2
    scores = w_ei * ei + w_ucb * ucb_term
    
    # Cost-aware scoring
    # Divide by cost^gamma
    gamma = 0.5
    scores = scores / np.power(cost_safe, gamma)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
