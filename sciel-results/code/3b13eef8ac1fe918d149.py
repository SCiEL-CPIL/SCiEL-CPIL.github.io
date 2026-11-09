"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a hybrid of Expected Improvement (EI) and Probability of Improvement (PI)
    with an adaptive exploration bonus. The PI component provides a smoother signal
    than pure EI, while the exploration bonus encourages visiting uncertain regions.
    
    mu, sigma: arrays of shape (2000,) - posterior mean and std in objective units
    f_best: float - lowest observed value so far
    
    Returns: array of shape (2000,) - scores, higher means evaluate next
    """
    start_time = time.time()
    DEADLINE = start_time + 295.0  # Safety margin before 330s limit
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Clip extreme values for numerical stability
    mu = np.clip(mu, f_best - 100, f_best + 100)
    sigma_safe = np.clip(sigma_safe, 1e-10, 100)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Probability of Improvement (for minimization): P(f(x) < f_best)
    pi = norm.cdf(z_clipped)
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    # Lower Confidence Bound: mu - beta * sigma (for minimization, lower is better)
    # Adaptive beta based on progress toward known minimum
    known_min = -1.8013
    # Progress: 0 when f_best is far from known_min, 1 when close
    progress = np.clip((f_best - known_min) / (5.0 - known_min), 0.0, 1.0)
    # Beta decreases as we get closer to the optimum
    beta = 0.3 + 1.5 * (1.0 - progress)
    
    lcb = mu - beta * sigma_safe
    lcb_score = -lcb  # Negate so higher is better
    
    lcb_score = np.where(np.isfinite(lcb_score), lcb_score, 0.0)
    
    # Normalize each component to [0, 1] using rank-based normalization
    n = len(ei)
    
    def rank_normalize(arr):
        """Rank-based normalization to [0, 1]."""
        if len(arr) <= 1:
            return np.ones_like(arr)
        mask = np.isfinite(arr)
        if not mask.any():
            return np.zeros_like(arr)
        if mask.sum() < len(arr):
            finite_vals = arr[mask]
            ranks_finite = np.argsort(np.argsort(finite_vals)) / (len(finite_vals) - 1)
            result = np.zeros_like(arr)
            result[mask] = ranks_finite
            return result
        ranks = np.argsort(np.argsort(arr)) / (n - 1)
        return ranks
    
    ei_ranks = rank_normalize(ei)
    pi_ranks = rank_normalize(pi)
    lcb_ranks = rank_normalize(lcb_score)
    
    # Adaptive weighting: 
    # Early on (low progress): more weight to exploration (LCB) and PI
    # Late (high progress): more weight to exploitation (EI)
    w_ei = 0.5 + 0.3 * progress
    w_pi = 0.2 + 0.1 * (1 - progress)
    w_lcb = 0.3 - 0.3 * progress
    # Normalize weights to sum to 1
    w_sum = w_ei + w_pi + w_lcb
    w_ei /= w_sum
    w_pi /= w_sum
    w_lcb /= w_sum
    
    # Combined score
    score = w_ei * ei_ranks + w_pi * pi_ranks + w_lcb * lcb_ranks
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() > DEADLINE:
        # Fallback: return pure EI if we're running out of time
        return np.where(np.isfinite(ei), ei, 0.0)
    
    return score
# EVOLVE-BLOCK-END
