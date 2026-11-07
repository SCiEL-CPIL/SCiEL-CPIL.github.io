"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive UCB.
    
    This function implements a structural change from the previous EI/UCB blend
    to a pure Upper Confidence Bound (UCB) approach with an adaptive exploration
    parameter (beta) that decays as the best observed value (f_best) improves.
    
    Parameters:
    mu: array of GP mean predictions (2000,)
    sigma: array of GP standard deviations (2000,)
    f_best: scalar, the best observed function value so far
    cost: array of costs for each candidate (2000,)
    
    Returns:
    score: array of acquisition scores (2000,), higher is better
    """
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Dynamic Beta:
    # In the early stages (high f_best), we want high exploration (high beta).
    # In the later stages (low f_best), we want low exploration (low beta).
    # We model this with an exponential decay of the exploration term.
    # f_best is typically positive and decreases towards 0.
    # Scale factor 5.0 is a heuristic: if f_best is ~5, beta is ~beta_min + (beta_max-beta_min)*0.37.
    beta_max = 2.5
    beta_min = 0.2
    decay_scale = 5.0
    
    # Calculate adaptive beta
    # If f_best is very large, exp(-large) is 0, so beta = beta_min? No, we want beta_max.
    # So: beta = beta_min + (beta_max - beta_min) * exp(-f_best / decay_scale)
    # If f_best -> 0, exp(0) = 1, beta = beta_max. This is backwards.
    # We want high beta when f_best is LARGE (early).
    # So: beta = beta_min + (beta_max - beta_min) * exp(-f_best / decay_scale) is wrong for "early = high f_best".
    # If f_best is 100, exp(-20) ~ 0. beta = beta_min.
    # We want beta = beta_max when f_best is 100.
    # So we should use: beta = beta_min + (beta_max - beta_min) * (1 - exp(-f_best / decay_scale))?
    # If f_best -> 0, 1 - 0 = 1. beta = beta_max. (Exploit? No, we want explore early).
    # Early: f_best is high. We want beta_max.
    # Late: f_best is low (0). We want beta_min.
    # So: beta = beta_min + (beta_max - beta_min) * exp(-f_best / decay_scale) is correct if we invert the logic?
    # Let's check:
    # f_best = 100. exp(-20) ~ 0. beta = beta_min. (Low exploration). BAD.
    # We want HIGH exploration early.
    # So beta should be HIGH when f_best is HIGH.
    # beta = beta_min + (beta_max - beta_min) * (1 - exp(-f_best / decay_scale))
    # f_best = 100 -> 1 - 0 = 1 -> beta = beta_max. (High exploration). GOOD.
    # f_best = 0 -> 1 - 1 = 0 -> beta = beta_min. (Low exploration). GOOD.
    
    beta = beta_min + (beta_max - beta_min) * (1.0 - np.exp(-f_best / decay_scale))
    
    # UCB for minimization: we want to minimize mu - beta*sigma.
    # So the acquisition score (to maximize) is beta*sigma - mu.
    ucb = beta * sigma_safe - mu
    
    # Cost-aware: divide by cost to prefer cheaper evaluations.
    # Use a power of 0.8 for a moderate discount.
    gamma = 0.8
    score = ucb / (cost_safe ** gamma)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
