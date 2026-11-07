"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a combination of Expected Improvement (EI)
    and a small UCB exploration term for robustness.
    Returns 2000 finite scores, higher meaning evaluate next.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small UCB term to encourage exploration
    # beta is a small exploration coefficient
    beta = 0.01
    ucb = mu + beta * sigma_safe
    
    # For minimization, we want to minimize ucb, so we negate it
    # Combine EI (which is positive for improvement) with -ucb (lower is better)
    # We need to scale them appropriately. 
    # Since EI is in objective units and ucb is in objective units, 
    # we can just add them with appropriate signs.
    # EI is already "higher is better". 
    # -ucb is "higher is better" (since lower ucb is better).
    
    # Let's just use EI with a slight modification to handle numerical stability
    # and add a tiny bit of exploration via the UCB term.
    
    # Actually, a common robust approach is to use EI but ensure that
    # points with very high uncertainty are not completely ignored.
    # The standard EI already does this via the sigma * pdf(z) term.
    
    # Let's try a weighted combination:
    # score = ei - alpha * ucb
    # But wait, EI is already a measure of expected improvement.
    # A pure EI is often optimal for minimization in BO.
    
    # Let's stick with EI but ensure numerical stability.
    # The key is to make sure we don't have NaNs or Infs.
    
    # One issue with standard EI is that when f_best is very good,
    # EI becomes very small everywhere, and the optimizer might get stuck.
    # Adding a small UCB term can help escape local optima.
    
    # Let's use a hybrid: EI + gamma * (-ucb)
    # But we need to be careful with scaling.
    
    # Actually, let's just use EI but with a slight twist:
    # We'll use the standard EI formula but ensure that if sigma is very large,
    # we don't over-penalize exploration.
    
    # Let's try: score = ei + epsilon * sigma
    # where epsilon is a small constant. This encourages exploration
    # in regions with high uncertainty.
    
    epsilon = 0.001
    score = ei + epsilon * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
