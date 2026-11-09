"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Quantile-based cost-aware acquisition function.
    
    Maximizes the expected reduction in the posterior lower quantile (10th percentile)
    of f(x), normalized by evaluation cost. This targets robust improvement against
    uncertainty, which is beneficial for navigating the Rosenbrock valley.
    
    mu, sigma: shape (2000,) GP mean and std in objective units
    f_best: lowest observed function value (scalar)
    cost: shape (2000,) evaluation cost for each candidate
    
    Returns: shape (2000,) acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Quantile parameter: 10th percentile means alpha = 0.1
    # The 10th percentile of f(x) is mu - z_0.9 * sigma, where z_0.9 = norm.ppf(0.9)
    # We want to maximize the reduction in this quantile from f_best.
    
    # z_0.9 is the 90th percentile of the standard normal (since we want the 10th percentile of f)
    # P(f(x) <= q) = 0.1 => (q - mu)/sigma = norm.ppf(0.1) => q = mu + norm.ppf(0.1) * sigma
    # norm.ppf(0.1) ≈ -1.2816
    
    alpha = 0.1
    z_alpha = norm.ppf(alpha)  # ≈ -1.2816
    
    # The 10th percentile of f(x) is: q(x) = mu + z_alpha * sigma
    q = mu + z_alpha * sigma_safe
    
    # The expected reduction in the quantile relative to f_best:
    # We want to maximize (f_best - q(x)) when q(x) < f_best, else 0
    # This is similar to Expected Improvement but on the quantile
    
    # Actually, a simpler and more robust approach:
    # Score = max(0, f_best - q(x)) / cost
    # But this can be zero for many points. Let's use a smoother version.
    
    # Alternative: Use the probability that the quantile beats f_best, combined with the magnitude
    # P(q(x) < f_best) = P(mu + z_alpha * sigma < f_best)
    # = P(z_alpha * sigma < f_best - mu)
    # = P(sigma < (f_best - mu) / z_alpha) ... this is getting complicated.
    
    # Let's stick with a simpler quantile-based score:
    # Score = (f_best - q(x)) / cost, clipped to be non-negative
    # This directly targets reducing the lower quantile.
    
    quantile_improvement = np.maximum(0, f_best - q)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = quantile_improvement / cost_safe
    
    # Add a small exploration term to avoid completely ignoring high-uncertainty regions
    # where the quantile might be low but the potential for discovery is high.
    # Exploration term: sigma / cost, scaled down
    exploration = 0.1 * sigma_safe / cost_safe
    
    scores = scores + exploration
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check (computation is O(N), should be very fast)
    if time.time() - start_time > 280.0:
         return np.zeros_like(mu)
         
    return scores
# EVOLVE-BLOCK-END
