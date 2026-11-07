"""Cost-aware acquisition for Powell 4D BO using Thompson Sampling.

Structural change: Switch from deterministic Probability of Improvement (PI) to Thompson
Sampling, a stochastic bandit-based acquisition. Instead of computing a fixed score for each
candidate, we sample a single global improvement threshold from the posterior uncertainty
distribution around f_best. Candidates are then scored by their probability of beating this
randomized threshold, which naturally balances exploration (via threshold variability) and
exploitation (via mu-driven probability) without explicit exploration hyperparameters.
The cost penalty is applied as a multiplicative discount to maintain cost-awareness.
"""
import numpy as np
from scipy.stats import norm
import time

# ------------------------------- deadline guard ---------------------------------
_T0 = time.time()
_DEADLINE = 330.0
_MARGIN = 15.0


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _T0)


# ------------------------------- public entry point ------------------------------
def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
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
    # Check deadline - if we're close to timeout, return a safe fallback
    if _time_left() < 1.0:
        # Emergency fallback: just return negative mu (exploitation)
        mu = np.asarray(mu, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1.0)

    # Thompson Sampling: Sample a stochastic improvement threshold from the posterior
    # The uncertainty in f_best is estimated from the spread of sigma values.
    # We model the threshold as f_best - delta, where delta ~ N(0, sigma_threshold^2)
    # sigma_threshold is derived from the median sigma (typical uncertainty scale)
    sigma_median = np.median(sigma)
    if not np.isfinite(sigma_median) or sigma_median < 1e-8:
        sigma_median = 1e-4
    
    # Sample the threshold offset: positive means we're looking for better (lower) values
    # The scale is proportional to the typical uncertainty, encouraging exploration
    # when uncertainty is high and exploitation when uncertainty is low
    threshold_offset = np.random.normal(0.0, sigma_median)
    threshold = f_best - threshold_offset
    
    # For each candidate, compute P(f(x) < threshold) = Phi((threshold - mu) / sigma)
    z = (threshold - mu) / sigma
    z = np.clip(z, -10, 10)  # Numerical stability
    
    prob_improve = norm.cdf(z)
    
    # Apply cost penalty: discount by cost to favor cheaper evaluations
    # Using a power penalty to balance exploration and cost efficiency
    cost_penalty = np.power(cost, 0.5)
    cost_safe = np.maximum(cost_penalty, 1e-6)
    
    score = prob_improve / cost_safe

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
