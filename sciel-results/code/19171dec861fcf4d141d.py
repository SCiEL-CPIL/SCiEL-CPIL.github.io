"""
Hybrid EI + UCB Acquisition Function with Standardization.

This program replaces the Knowledge Gradient approach with a robust hybrid of
Expected Improvement (EI) and Upper Confidence Bound (UCB).

Key structural changes:
1. Abandons Monte Carlo KG for deterministic EI and UCB.
2. Standardizes both EI and UCB scores (zero mean, unit variance) to ensure
   fair blending regardless of the scale of the objective function.
3. Uses an adaptive blending weight based on relative uncertainty to balance
   exploration and exploitation.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 15.0  # seconds

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a standardized hybrid of Expected Improvement (EI)
    and Upper Confidence Bound (UCB).
    
    Args:
        mu: Mean of the GP posterior at candidates (shape 2000,).
        sigma: Std dev of the GP posterior at candidates (shape 2000,).
        f_best: Current best function value (minimum).
        
    Returns:
        scores: Array of shape 2000, higher is better.
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback to simple EI if too close to deadline
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    n = len(mu)
    
    # Ensure inputs are finite and sigma is positive
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # 1. Compute Expected Improvement (EI)
    # EI = E[max(f_best - Y, 0)]
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in cdf/pdf, though scipy handles large values well
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute Upper Confidence Bound (UCB)
    # For minimization, we want to explore regions with low mu and high sigma.
    # UCB score = -mu + beta * sigma
    # We use a fixed beta for stability, as adaptive beta can be sensitive to noise.
    # A beta of 1.0 is a standard starting point for 1D problems.
    beta = 1.0
    ucb = -mu + beta * sigma
    
    # 3. Standardize both scores to zero mean and unit variance
    # This ensures that the blending weights have a consistent effect.
    
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std < 1e-9:
        ei_std = 1e-9
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb)
    if ucb_std < 1e-9:
        ucb_std = 1e-9
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # 4. Determine blending weight
    # We want to explore more when uncertainty is high relative to the current best.
    # rel_uncertainty = mean(sigma) / (|f_best| + 1)
    # If f_best is close to 0, this might be large. If f_best is large negative, it might be small.
    # Let's use a simpler heuristic:
    # If the mean sigma is large compared to the range of mu, we are in an exploratory phase.
    
    # Alternative: Use a fixed weight that has been shown to work well (0.5/0.5 or 0.6/0.4).
    # Inspiration 2 used an adaptive weight based on tanh of relative uncertainty.
    # Let's try a robust adaptive weight:
    
    # Measure relative uncertainty:
    # If sigma is large everywhere, we should explore more.
    # If sigma is small, we should exploit.
    
    # A simple proxy: the ratio of mean sigma to the standard deviation of mu.
    # If sigma is comparable to the variation in mu, exploration is valuable.
    
    mu_std = np.std(mu)
    if mu_std < 1e-9:
        mu_std = 1e-9
        
    rel_unc = np.mean(sigma) / mu_std
    
    # Map rel_unc to a weight for UCB between 0.3 and 0.7
    # If rel_unc is very low (exploit), w_ucb should be low.
    # If rel_unc is high (explore), w_ucb should be high.
    
    # Use a smooth transition
    # w_ucb = 0.5 + 0.2 * tanh(rel_unc - 1.0)
    # This keeps it in [0.3, 0.7] roughly.
    
    w_ucb = 0.5 + 0.2 * np.tanh(rel_unc - 1.0)
    w_ei = 1.0 - w_ucb
    
    # 5. Combine scores
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # 6. Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
