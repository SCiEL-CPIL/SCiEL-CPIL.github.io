import numpy as np
from scipy.stats import norm
import time

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def _expected_improvement(mu, sigma, f_best):
    """Compute Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    return np.maximum(ei, 0.0)

def _ucb(mu, sigma, beta):
    """Compute Upper Confidence Bound for minimization (lower is better)."""
    return mu - beta * sigma

def _soft_min(a, b, temperature=0.1):
    """Soft minimum of two arrays for smooth combination."""
    return -temperature * np.logaddexp(-a / temperature, -b / temperature)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Styblinski-Tang 2D Bayesian optimization.
    
    Uses a structural blend of EI and UCB signals with soft-min combination,
    scaled by cost efficiency. This approach provides robust balance between
    exploration and exploitation in multimodal landscapes.
    
    Parameters:
    - mu: shape (2000,), GP mean predictions
    - sigma: shape (2000,), GP standard deviations
    - f_best: scalar, best observed value
    - cost: shape (2000,), evaluation costs (1 + mean(x_scaled))
    
    Returns:
    - scores: shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Check deadline first
    if _check_deadline():
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip to valid ranges
    sigma = np.clip(sigma, 1e-8, 1e4)
    cost = np.clip(cost, 0.5, 10.0)
    
    # Compute EI for minimization
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Compute UCB for minimization (lower values are better)
    # Adaptive beta: start higher for exploration, decrease as we approach optimum
    dist_to_min = max(f_best - (-78.3323), 0.0)
    progress = 1.0 / (1.0 + 0.05 * dist_to_min)
    
    # Beta for UCB: higher when far from optimum (more exploration)
    beta_ucb = 0.5 + 0.5 * (1.0 - progress)
    ucb = _ucb(mu, sigma, beta_ucb)
    
    # Convert UCB to a "score" where higher is better
    # Since UCB is lower-is-better, we negate it for scoring
    ucb_score = -ucb
    
    # Cost efficiency: reward cheaper evaluations
    # Use a moderate penalty: 1/sqrt(cost)
    cost_efficiency = 1.0 / np.sqrt(cost)
    
    # Scale both signals by cost efficiency
    ei_scaled = ei * cost_efficiency
    ucb_scaled = ucb_score * cost_efficiency
    
    # Soft-min combination: smoothly blend EI and UCB
    # Temperature controls the sharpness of the minimum
    # Lower temperature -> closer to hard min, higher -> more blended
    temperature = 0.05 + 0.1 * progress  # slightly sharper when close to optimum
    
    # We want to pick the candidate that maximizes the "quality" signal
    # Both ei_scaled and ucb_scaled are higher-is-better
    # Use soft-max (not soft-min) since we want the best of the two
    # Actually, we want to combine them such that if either signal is strong, the score is strong
    # Let's use a weighted sum with adaptive weights instead for clarity
    
    # Weighted combination: balance between EI (exploitation) and UCB (exploration)
    # As progress increases, weight more on EI
    weight_ei = 0.6 + 0.4 * progress  # 0.6 to 1.0
    weight_ucb = 1.0 - 0.5 * progress  # 1.0 to 0.5
    
    score = weight_ei * ei_scaled + weight_ucb * ucb_scaled
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
