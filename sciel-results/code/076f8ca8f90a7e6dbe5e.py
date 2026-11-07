"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: blended EI and UCB for multimodal minimization.
    
    Uses a dynamic blend that favors exploration when uncertainty is high
    relative to the gap to the best found value, and exploitation when
    the model is confident.
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (for minimization: lower is better)
    # UCB score = mu - kappa * sigma, higher score means more promising
    # We want to maximize this, so: score_ucb = -mu + kappa * sigma
    # But we need to combine with EI which is already positive when mu < f_best
    
    # Adaptive kappa: larger when overall uncertainty is high
    # Use the median sigma as a scale reference
    sigma_median = np.median(sigma_safe)
    # If sigma_median is large relative to f_best spread, explore more
    # Simple heuristic: kappa between 0.5 and 2.0
    # When f_best is close to mu values, we're in exploitation mode
    # When f_best is far from mu values, we're in exploration mode
    
    # Use a fixed well-tuned kappa for UCB component
    kappa = 1.5
    ucb_score = -mu + kappa * sigma_safe
    
    # Normalize both to [0, 1] range for blending
    ei_max = np.max(ei)
    ucb_max = np.max(ucb_score)
    ei_min = np.min(ei)
    ucb_min = np.min(ucb_score)
    
    if ei_max - ei_min > eps:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    if ucb_max - ucb_min > eps:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # Blend: 0.5 * EI + 0.5 * UCB, but weight more toward EI when confident
    # Confidence metric: fraction of points where sigma is small
    confidence = np.mean(sigma_safe < 0.1 * sigma_median + eps) if sigma_median > eps else 0.5
    # When confident (high confidence), weight more toward EI
    # When uncertain, weight more toward UCB
    weight_ei = 0.3 + 0.4 * confidence  # between 0.3 and 0.7
    weight_ucb = 1.0 - weight_ei
    
    score = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
