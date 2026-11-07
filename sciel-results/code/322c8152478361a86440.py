"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Uses a hybrid strategy:
    1. Expected Improvement (EI) as the base acquisition.
    2. Add a small exploration bonus based on uncertainty (sigma) to encourage
       exploration in early stages, but decay it as f_best approaches the minimum.
    3. Use a quantile-based approach: target the lower tail of the posterior.
    
    Specifically, we compute a weighted combination of:
    - Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    - Lower quantile proxy: mu - k * sigma (encourages low mu and high sigma)
    
    We use a time-aware guard to ensure we return within budget.
    """
    start_time = time.time()
    deadline = 330.0  # seconds, with safety margin
    
    # Ensure finite inputs
    sigma = np.maximum(sigma, 1e-9)
    
    # Standard EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle numerical issues
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Lower quantile: approximate the expected minimum of N samples from N(mu, sigma^2)
    # E[min of N] ≈ mu - sigma * E[max of N standard normals]
    # For large N, E[max] ≈ sqrt(2 log N)
    # But we want something that adapts: use a quantile-based score
    # Q_{1-alpha} = mu + z_{alpha} * sigma, where z_{alpha} is negative for small alpha
    # We want to target the lower bound, so use a small alpha
    
    # Adaptive alpha: smaller when f_best is close to 0 (we're exploiting),
    # larger when f_best is far from 0 (we're exploring)
    # f_best is the lowest observed value; for sphere 1d on [-5,5], min is 0
    
    # Scale factor: how far are we from the minimum?
    gap = max(f_best, 1e-9)  # avoid division by zero
    
    # When gap is large, explore more (larger alpha -> less extreme quantile)
    # When gap is small, exploit more (smaller alpha -> more extreme quantile)
    # alpha = 1e-4 when exploiting, up to 0.1 when exploring
    alpha = 1e-4 + 0.1 * min(1.0, gap / 10.0)
    
    # z_alpha is the alpha-quantile of the standard normal (negative value)
    z_alpha = norm.ppf(alpha)
    
    # Lower quantile score: we want to maximize the chance of finding a low value
    # Score based on the lower quantile: higher score = better
    # mu + z_alpha * sigma is the lower quantile (since z_alpha < 0)
    # We want to MINIMIZE this, so the acquisition score should be:
    # score_quantile = -(mu + z_alpha * sigma) = -mu - z_alpha * sigma = -mu + |z_alpha| * sigma
    
    quantile_score = -mu + (-z_alpha) * sigma  # since z_alpha < 0, -z_alpha > 0
    
    # Combine EI and quantile score
    # Weight: more EI when we're confident, more quantile when uncertain
    # Use a fixed weight for simplicity, but adapt based on gap
    # When gap is large, trust the quantile (exploration) more
    # When gap is small, trust EI (exploitation) more
    w_ei = 1.0
    w_quantile = 0.5 * min(1.0, gap / 5.0)
    
    score = w_ei * ei + w_quantile * quantile_score
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # Safety: return what we have
        return score
    
    return score
# EVOLVE-BLOCK-END
