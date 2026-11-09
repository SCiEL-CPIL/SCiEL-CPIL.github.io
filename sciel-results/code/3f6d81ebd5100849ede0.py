"""Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture in 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: combines Expected Improvement (EI) with a UCB-style exploration term.
    The exploration term is scaled by a coefficient that decays as the posterior uncertainty
    shrinks, encouraging exploration early and exploitation later.
    """
    # Safety: ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB-style exploration term
    # Use a moderate beta that balances exploration and exploitation
    # The exploration term is proportional to sigma, encouraging high-uncertainty regions
    beta = 1.5
    ucb = mu + beta * sigma
    
    # Since we are minimizing, lower mu + beta*sigma is better
    # But we want higher scores to be better, so we negate the UCB term
    # Combine EI (which is already in "higher is better" form) with a transformed UCB
    # The UCB term as a score: -ucb (since lower ucb is better for minimization)
    # But we need to balance scales. EI is in objective units, -ucb is also in objective units.
    # Let's use a weighted sum. EI tends to be small when close to optimum, UCB term can be larger.
    # Scale the UCB contribution by a factor that depends on the current uncertainty level.
    
    # A simpler approach: use a PI-like term that gives high scores to points with low mean
    # and high uncertainty, but bounded.
    
    # Let's use: score = ei + alpha * sigma, where alpha is a small positive constant
    # This adds a small exploration bonus proportional to uncertainty.
    alpha = 0.1
    score = ei + alpha * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
