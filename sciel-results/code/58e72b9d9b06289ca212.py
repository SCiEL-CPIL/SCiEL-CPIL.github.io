"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _expected_improvement(mu, sigma, f_best):
    """Compute EI with numerical stability."""
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in extreme tails
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    # Where sigma is very small and mu < f_best, EI should be f_best - mu
    # Where sigma is very small and mu >= f_best, EI should be 0
    small_sigma = sigma < 1e-8
    ei[small_sigma] = np.maximum(f_best - mu[small_sigma], 0.0)
    return np.maximum(ei, 0.0)


def _ucb(mu, sigma, f_best, xi=0.0):
    """Compute UCB with cost awareness built in externally."""
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    # Use a dynamic kappa based on the spread of sigma
    kappa = 1.0
    return mu + kappa * sigma - xi


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines multiple acquisition strategies with cost awareness.
    Uses a weighted combination of:
    1. Expected Improvement per unit cost
    2. UCB per unit cost  
    3. A hybrid that accounts for uncertainty in cost
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    n = len(mu)
    
    # Ensure inputs are finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 0.1)  # Ensure positive cost
    
    # Compute EI
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Compute UCB
    ucb = _ucb(mu, sigma, f_best, xi=0.0)
    
    # Cost-normalized scores
    ei_per_cost = ei / cost
    ucb_per_cost = ucb / cost
    
    # Additional term: penalize high cost more aggressively for exploration
    # Use a logarithmic penalty to avoid over-penalizing
    cost_penalty = np.log1p(cost)
    
    # Combine strategies with adaptive weighting
    # When sigma is high (exploration), weight UCB more
    # When sigma is low (exploitation), weight EI more
    sigma_norm = sigma / (np.max(sigma) + 1e-12)
    weight_ucb = 0.3 + 0.4 * sigma_norm  # 0.3 to 0.7
    weight_ei = 1.0 - weight_ucb
    
    # Base score
    base_score = weight_ei * ei_per_cost + weight_ucb * ucb_per_cost
    
    # Add a small exploration bonus for points with higher uncertainty
    # This helps escape local optima
    exploration_bonus = 0.1 * sigma / (np.max(sigma) + 1e-12)
    
    # Final score with cost penalty
    score = base_score + exploration_bonus - 0.05 * cost_penalty
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 270.0:  # If we're getting close to the deadline
        # Return a simpler, faster score
        simple_score = ei / cost
        simple_score = np.where(np.isfinite(simple_score), simple_score, 0.0)
        return simple_score
    
    return score
# EVOLVE-BLOCK-END
