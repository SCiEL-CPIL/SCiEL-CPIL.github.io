"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-adjusted Lower Confidence Bound (LCB) acquisition function.
    
    Maximizes -(mu + kappa * sigma) / cost to target the global minimum directly,
    with dynamic kappa based on remaining budget fraction.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Dynamic kappa: larger when more budget remains (explore more),
    # smaller when budget is tight (exploit more).
    # Total budget is 30, initial cost from 5 points is roughly 5 * 1.5 = 7.5
    # So remaining fraction goes from ~0.75 to 0.
    remaining_fraction = np.clip((30.0 - np.mean(cost)) / 30.0, 0.0, 1.0)
    
    # kappa scales from ~2.5 (high exploration) to ~0.5 (low exploration)
    kappa = 0.5 + 2.0 * remaining_fraction
    
    # LCB: minimize mu + kappa * sigma, so we negate for maximization
    lcb = mu + kappa * sigma
    
    # Cost-adjusted: divide by cost (higher cost means less attractive)
    score = -lcb / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
