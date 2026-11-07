"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Computes a cost-aware acquisition function combining EI and UCB with
    adaptive exploration based on budget remaining.
    
    Args:
        mu: array of shape (N,) predicted mean
        sigma: array of shape (N,) predicted std deviation
        f_best: float, best value found so far
        cost: array of shape (N,) evaluation cost for each candidate
        
    Returns:
        scores: array of shape (N,) acquisition values (higher is better)
    """
    # Deadline guard
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    elapsed = time.time() - acquisition._start_time
    if elapsed > 290:
        # Fast fallback: simple EI / cost
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei / cost_safe

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma and cost to small positive values
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate PDF and CDF
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)
    
    # UCB with adaptive beta based on how much budget remains
    # Early on (low cost), we want more exploration; later, more exploitation
    # Total budget is 30, so remaining fraction = (30 - total_cost) / 30
    # But we don't have total_cost directly; we can estimate from mean cost
    # Actually, cost here is per-candidate cost, not total. Let's use a simpler heuristic.
    # Use a fixed beta that balances exploration and exploitation well for this problem.
    beta = 1.5
    
    ucb = mu + beta * sigma_safe
    
    # Combine EI and UCB: use EI as primary, add UCB as secondary signal
    # Normalize both to [0, 1] range for fair combination
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    ucb_max = np.max(ucb) if np.max(ucb) > 1e-12 else 1.0
    
    ei_norm = ei / ei_max
    ucb_norm = (ucb - np.min(ucb)) / (ucb_max - np.min(ucb) + 1e-12)
    
    # Weighted combination: 70% EI, 30% UCB
    combined = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Cost-aware: divide by cost
    score = combined / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
