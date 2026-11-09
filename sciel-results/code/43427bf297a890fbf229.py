"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid approach combining:
    1. Expected Improvement (EI) scaled by a cost-aware factor
    2. A lower confidence bound (LCB) term weighted by cost
    3. A small exploration bonus for high-uncertainty regions
    
    The cost model is 1 + mean(x_scaled), where x_scaled is in [0,1].
    We use the provided cost array directly.
    
    Strategy: Use a weighted combination of cost-normalized EI and LCB,
    with adaptive weights based on the relationship between f_best and the
    mean predictions. This helps balance exploitation and exploration
    while accounting for evaluation costs.
    """
    eps = 1e-9
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement (EI)
    # Standard EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in norm.cdf
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # LCB (Lower Confidence Bound) - we want to minimize, so lower is better
    # LCB = mu - beta * sigma
    # For acquisition (higher = more promising), we use -LCB = -mu + beta * sigma
    # But we need to be careful: we want to find low values
    # So acquisition should be high where mu is low and sigma is high
    
    # Cost-aware weighting
    # Normalize cost to [0, 1] range for weighting
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + eps
    cost_norm = (cost - cost_min) / cost_range  # 0 = cheapest, 1 = most expensive
    
    # Weight 1: Cost-normalized EI
    # Higher EI per unit cost is better
    ei_cost = ei / (cost + eps)
    
    # Weight 2: LCB-based term with cost penalty
    # Use a beta that balances exploration and exploitation
    # Adaptive beta: larger when we're far from the minimum, smaller when close
    # Estimate progress: how much have we improved from a rough baseline
    # Since we don't know the true minimum, use relative measure
    lcb = mu - 2.0 * sigma_safe  # Lower bound on the objective
    # For acquisition: we want high scores for low lcb and low cost
    # Use: -lcb / cost, which rewards low predicted values and low cost
    lcb_cost = -lcb / (cost + eps)
    
    # Weight 3: Pure exploration for high-uncertainty regions
    # With a mild cost penalty to avoid expensive exploration
    exploration = sigma_safe / np.sqrt(cost + eps)
    
    # Combine terms with fixed weights
    # EI per cost is the primary driver
    # LCB per cost helps find genuinely low-value regions
    # Exploration ensures we don't get stuck
    score = 0.7 * ei_cost + 0.2 * lcb_cost + 0.1 * exploration
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
