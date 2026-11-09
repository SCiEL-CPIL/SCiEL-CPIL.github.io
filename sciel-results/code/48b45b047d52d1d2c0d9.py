"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Uses a pure cost-normalized LCB approach:
    score = -(mu - kappa * sigma) / cost
    
    This directly optimizes the predicted lower bound per unit cost,
    which is the natural objective for cost-aware Bayesian optimization.
    Higher scores indicate points with lower predicted values and/or
    higher uncertainty, weighted by their evaluation cost.
    """
    eps = 1e-9
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid issues
    sigma_safe = np.maximum(sigma, eps)
    
    # Cost-aware LCB: we want to minimize mu - kappa*sigma, but also minimize cost
    # So we maximize (f_best - (mu - kappa*sigma)) / cost, or equivalently
    # minimize (mu - kappa*sigma) / cost, which means maximizing -(mu - kappa*sigma) / cost
    
    # Use a fixed kappa for exploration-exploitation balance
    kappa = 1.0
    
    # LCB value (lower is better)
    lcb = mu - kappa * sigma_safe
    
    # Score: higher is better, so we negate the LCB and normalize by cost
    score = -lcb / (cost + eps)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
