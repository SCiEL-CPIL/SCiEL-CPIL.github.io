"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP in objective units.
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP in objective units.
    f_best : float
        Best observed value so far (lowest since we're minimizing).
    cost : array-like, shape (2000,)
        Cost of evaluating each point, shape (2000,).
        Cost = 1 + mean(x scaled to [0, 1]) where x is in [-10, 10]^2.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values (higher = evaluate next).
    """
    # Deadline guard
    t0 = time.time()
    deadline = 295  # safety margin before 300s limit
    
    # Ensure arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    cdf_vals = norm.cdf(z_clipped)
    pdf_vals = norm.pdf(z_clipped)
    
    # EI for minimization
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Cost-aware: EI per unit cost
    scores_ei = ei / cost_safe
    
    # Thompson Sampling: sample a single objective value from the Gaussian posterior
    # y ~ N(mu, sigma)
    # Use a fixed random state for reproducibility within a single call
    # but since we can't seed per call easily without affecting global state,
    # we use a local generator
    rng = np.random.default_rng()
    eps = rng.standard_normal(mu.shape)
    y_ts = mu + sigma_safe * eps
    
    # Cost-aware Thompson Sampling score:
    # We want to minimize y, so higher score means lower y.
    # score = -y / cost
    scores_ts = -y_ts / cost_safe
    
    # Combine EI and TS scores
    # Use a weighted sum. The weight can be tuned, but let's start with equal weight
    # or a bias towards EI which is more deterministic.
    # Actually, TS can be too stochastic. Let's blend them.
    # A common approach is to use EI as the primary signal and TS as an exploration bonus.
    # Or, normalize both to have similar scales.
    
    # Let's normalize both scores to zero mean and unit variance for fair combination
    def normalize(arr):
        m = np.mean(arr)
        s = np.std(arr)
        if s < 1e-12:
            return np.zeros_like(arr)
        return (arr - m) / s
    
    scores_ei_norm = normalize(scores_ei)
    scores_ts_norm = normalize(scores_ts)
    
    # Weight: 0.7 for EI (exploitation), 0.3 for TS (exploration)
    w_ei = 0.7
    w_ts = 0.3
    
    scores = w_ei * scores_ei_norm + w_ts * scores_ts_norm
    
    # Add a small exploration bonus based on uncertainty to ensure coverage
    # sigma_norm = sigma_safe / (np.max(sigma_safe) + 1e-12)
    # exploration_weight = 0.05
    # scores = scores + exploration_weight * sigma_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time check
    if time.time() - t0 > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
