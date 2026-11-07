"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI) and
    Lower Confidence Bound (LCB) with dynamic exploration/exploitation balance.
    
    Args:
        mu: Array of shape (N,) with GP mean predictions.
        sigma: Array of shape (N,) with GP standard deviations.
        f_best: Scalar, the best observed function value so far.
        cost: Array of shape (N,) with evaluation costs for each candidate.
        
    Returns:
        Array of shape (N,) with scores. Higher score is better.
    """
    start_time = time.time()
    deadline = start_time + 295.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    f_best_val = float(f_best)
    N = len(mu)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Dynamic beta for LCB: more exploration when f_best is worse
    # f_best is typically in [0, 1] for Griewank
    f_best_clipped = np.clip(f_best_val, 0.0, 1.0)
    beta_lcb = 1.0 + 2.0 * f_best_clipped
    
    # Calculate EI (Expected Improvement) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal
    z = (f_best_val - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Clip for numerical stability
    ei = (f_best_val - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Calculate LCB score (higher is better for "optimistic" minimization)
    # LCB = mu - beta * sigma, we want low LCB, so score = -LCB = -mu + beta*sigma
    lcb_score = -mu + beta_lcb * sigma_safe
    
    # Dynamic blend: use more EI when close to optimum, more LCB when exploring
    # weight_ei increases as f_best gets better (closer to 0)
    weight_ei = 1.0 - f_best_clipped  # 1.0 when f_best=0, 0.0 when f_best=1
    weight_lcb = 1.0 - weight_ei
    
    # Combine EI and LCB scores
    # Normalize both to comparable scales
    # EI is in objective units, LCB score is also in objective units
    combined_score = weight_ei * ei + weight_lcb * lcb_score
    
    # Cost-adjusted score: divide by cost to penalize expensive evaluations
    scores = combined_score / cost_safe
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: simple exploitation
        scores = -mu / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
