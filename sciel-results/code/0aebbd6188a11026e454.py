"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d).
Strategy: Cost-aware Gaussian Process-UCB with adaptive beta and a small EI floor.
The primary score is (f_best - mu + beta * sigma) / cost, which is a UCB-style
acquisition that strongly rewards uncertainty. A small EI component is added
to ensure that points with high expected improvement are not completely ignored.
The beta parameter adapts based on the number of observations (estimated from
the spread of sigma) to balance exploration and exploitation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to spare

def _is_deadline_exceeded():
    """Check if we are close to the global time limit."""
    if time.time() - _START_TIME > (300.0 - _DEADLINE_MARGIN):
        return True
    return False

def _expected_improvement(mu, sigma, f_best):
    """
    Compute Expected Improvement for minimization.
    EI(x) = E[max(f_best - f(x), 0)]
    where f(x) ~ N(mu(x), sigma(x)^2)
    """
    sigma_safe = np.clip(sigma, 1e-12, None)
    improvement = f_best - mu
    z = improvement / sigma_safe
    # EI = improvement * Phi(z) + sigma * phi(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI should be non-negative
    return np.clip(ei, 0.0, None)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB with adaptive beta.
    
    Parameters:
    mu: array-like, mean predictions from GP
    sigma: array-like, standard deviation predictions from GP
    f_best: float, best observed value so far
    cost: array-like, cost of evaluating each candidate
    
    Returns:
    scores: array of finite scores, higher is better
    """
    # Safety check: if time is running out, return a safe random-like score
    # to allow the loop to finish or fail gracefully without hanging.
    if _is_deadline_exceeded():
        return np.zeros_like(mu, dtype=float)

    # Ensure inputs are numpy arrays for vectorization
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Numerical stability
    epsilon = 1e-8
    sigma_safe = np.clip(sigma, epsilon, None)
    cost_safe = np.clip(cost, epsilon, None)
    
    # Estimate the number of observations from the spread of sigma
    # In GP, sigma is high in unexplored regions and low near observations.
    # The mean sigma can give a rough sense of how much of the domain is explored.
    mean_sigma = np.mean(sigma_safe)
    
    # Heuristic: estimate n_obs from mean_sigma
    # Initially, sigma is high (around the prior std, which is 1 after standardization).
    # As we add points, sigma decreases near those points.
    # A rough model: mean_sigma ~ 1/sqrt(n_obs) for a well-spread set of points.
    # So n_obs ~ 1/mean_sigma^2
    # But this is very rough. Let's use a simpler adaptive scheme.
    
    # Adaptive beta: start high for exploration, decrease for exploitation
    # Use a logarithmic decay based on estimated observations
    # Estimate n_obs: if mean_sigma is close to 1, n_obs is small. If mean_sigma is small, n_obs is large.
    # Let's use n_obs_est = max(5, int(1 / (mean_sigma + 0.1)))
    n_obs_est = max(5, int(1 / (mean_sigma + 0.1)))
    
    # Beta decay: beta = 2.0 / log(n_obs_est + 1)
    # At n=5, log(6)~1.8, beta~1.1
    # At n=20, log(21)~3.0, beta~0.67
    # At n=50, log(51)~3.9, beta~0.51
    beta = 2.0 / np.log(n_obs_est + 1)
    
    # Ensure beta is within a reasonable range
    beta = np.clip(beta, 0.3, 2.5)
    
    # UCB score: (f_best - mu + beta * sigma) / cost
    # This is the primary score, emphasizing exploration via sigma
    ucb_score = (f_best - mu + beta * sigma_safe) / cost_safe
    
    # EI score: small weight to ensure we don't completely ignore exploitation
    ei_scores = _expected_improvement(mu, sigma_safe, f_best)
    ei_score_normalized = ei_scores / cost_safe
    
    # Blend: primarily UCB, with a small EI component
    # The weight of EI can also adapt: higher when beta is lower (later stages)
    w_ei = 0.1 + 0.1 * (1.0 - beta / 2.5)  # Between 0.1 and 0.2
    
    combined_score = (1.0 - w_ei) * ucb_score + w_ei * ei_score_normalized
    
    # Final safety: replace any non-finite values with 0
    scores = np.where(np.isfinite(scores := combined_score), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
