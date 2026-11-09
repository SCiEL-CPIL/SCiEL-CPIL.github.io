# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for minimizing hartmann 3d.

    Uses Best-of-N Thompson Sampling.
    For each candidate, we draw N samples from the GP posterior N(mu, sigma^2).
    We take the best (lowest) observed value in these samples as an estimate
    of the potential minimum. We then divide by cost to get a cost-efficiency score.
    This is a stochastic acquisition function that naturally balances exploration
    (high sigma leads to potentially very low samples) and exploitation
    (low mu leads to consistently low samples).
    """
    t0 = time.time()
    deadline = t0 + 280.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    n = len(mu_safe)
    if n == 0:
        return np.array([])

    # Number of samples for Thompson Sampling
    # 10-20 is a good balance between cost and accuracy
    n_samples = 20
    
    # Initialize best scores (we want to minimize objective, so start with high values)
    # We are computing -objective / cost. So we want to MAXIMIZE this.
    # Equivalent to minimizing objective / cost.
    # Let's compute the best (lowest) objective value from samples, then convert to score.
    
    # To be efficient, we can draw all samples at once
    # Shape: (n_samples, n)
    # samples = mu_safe[None, :] + sigma_safe[None, :] * np.random.randn(n_samples, n)
    
    # However, doing this in one go uses memory: 20 * 2000 * 8 bytes = 320KB. Negligible.
    
    if time.time() > deadline:
        # Fallback to simple EI if too slow (unlikely)
        z = (f_best - mu_safe) / sigma_safe
        z = np.clip(z, -10, 10)
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        ei = (f_best - mu_safe) * cdf_z + sigma_safe * pdf_z
        ei = np.maximum(ei, 0.0)
        return ei / cost_safe

    # Draw samples
    # Use a fixed seed? No, stochasticity is good for exploration across iterations.
    # But for reproducibility within a run, it doesn't matter as long as it's valid.
    
    # Optimization: Instead of storing all samples, we can update the "best so far"
    # iteratively to save memory and potentially allow early stopping?
    # But N is small. Let's just do vectorized.
    
    # samples = mu_safe[None, :] + sigma_safe[None, :] * np.random.randn(n_samples, n)
    # best_obj = np.min(samples, axis=0)
    # scores = -best_obj / cost_safe
    
    # Wait, the task says "Return 2000 finite scores (higher = evaluate next)".
    # If we use -best_obj / cost, lower obj (better) -> higher score.
    # This is correct.
    
    # One issue: Thompson Sampling samples are from the *posterior predictive*.
    # The "best of N" is an estimate of the quantile.
    # Is this better than EI?
    # EI = E[max(0, f_best - f(x))].
    # Best-of-N TS approximates the expected value of the minimum of N draws.
    # As N -> infinity, Best-of-N TS -> inf (if sigma > 0).
    # So we must normalize or limit N.
    # With N=20, it's a specific quantile.
    
    # Let's try N=10 to be conservative.
    n_samples = 10
    samples = mu_safe[None, :] + sigma_safe[None, :] * np.random.randn(n_samples, n)
    best_obj = np.min(samples, axis=0)
    
    # Convert to score
    # We want to prefer points with low best_obj and low cost.
    scores = -best_obj / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
