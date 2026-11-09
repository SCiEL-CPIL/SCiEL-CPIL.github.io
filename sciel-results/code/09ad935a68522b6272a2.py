"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a blended, cost-aware acquisition score for 2000 candidates.
    Blends Expected Improvement (EI) and Upper Confidence Bound (UCB) per unit cost.
    """
    # Ensure finite inputs; replace non-finite with safe defaults
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)

    # Clip sigma to avoid division by zero and numerical issues
    sigma = np.clip(sigma, 1e-9, None)

    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.clip(ei, 0.0, None)

    # Upper Confidence Bound (UCB) with beta tuned for exploration
    beta = 2.0
    ucb = mu + beta * sigma

    # We want to minimize, so lower mu is better. UCB for minimization:
    # We prefer points with low mu and high uncertainty.
    # Actually for minimization, a common UCB is mu - beta*sigma (lower bound of confidence).
    # But in acquisition functions for minimization, we often use:
    #   acquisition = -mu + beta*sigma  (we want high acquisition = low mu, high sigma)
    # Or equivalently, we can use EI which already handles this.
    # Let's define a "minimization UCB" score where higher is better:
    #   ucb_score = -mu + beta * sigma
    ucb_score = -mu + beta * sigma

    # Normalize both to comparable scales before blending
    # Use a simple approach: weight EI and UCB_score
    # EI is in objective units, ucb_score is also in objective units (roughly)
    
    # Blend: 70% EI, 30% UCB for balanced exploration/exploitation
    blended = 0.7 * ei + 0.3 * ucb_score

    # Cost-aware: divide by cost
    score = blended / cost

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
