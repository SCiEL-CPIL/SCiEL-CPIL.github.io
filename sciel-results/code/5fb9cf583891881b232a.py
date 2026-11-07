"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Hybrid acquisition: Expected Improvement + adaptive exploration bonus
    # EI naturally scales with potential improvement; the bonus decays smoothly.
    
    # Compute Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # Handle the case where sigma is very small
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)  # Clip to avoid numerical issues
    
    EI = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    EI = np.maximum(EI, 0.0)  # EI should be non-negative
    
    # Adaptive exploration bonus:
    # Use a bonus that scales with sigma but decays as we get more confident.
    # The key insight from the improved solution: use a noise-aware exploration bonus
    # that is derived from the GP posterior and decays smoothly.
    
    # Compute a scale for the exploration bonus based on the current uncertainty
    # Use the median sigma to get a robust scale estimate
    sigma_median = np.median(sigma_safe)
    
    # Exploration bonus: proportional to sigma, but scaled down as the search progresses
    # We use a simple heuristic: the bonus is beta * sigma, where beta adapts
    # based on the ratio of the best possible improvement to the current uncertainty
    
    # The maximum possible improvement from any point is roughly (max(mu) - f_best)
    # But we don't know max(mu) directly. Instead, we use the spread of mu.
    mu_spread = np.max(mu) - np.min(mu)
    if mu_spread < eps:
        mu_spread = sigma_median + eps
    
    # Adaptive beta: starts high when uncertainty is large relative to improvement,
    # decays as we converge
    # beta = c * (sigma_median / (mu_spread + sigma_median))
    # This gives beta ~ 0.5*c when uncertainty dominates, and ~ 0 when improvement dominates
    c = 2.0  # Tuning constant
    beta = c * (sigma_median / (mu_spread + sigma_median + eps))
    
    # Clip beta to a reasonable range
    beta = np.clip(beta, 0.1, 3.0)
    
    # Exploration bonus
    exploration_bonus = beta * sigma_safe
    
    # Combine EI and exploration bonus
    # Normalize both to comparable scales before combining
    EI_max = np.max(EI)
    if EI_max < eps:
        EI_max = 1.0
    
    bonus_max = np.max(exploration_bonus)
    if bonus_max < eps:
        bonus_max = 1.0
    
    # Normalize to [0, 1] range
    EI_norm = EI / EI_max
    bonus_norm = exploration_bonus / bonus_max
    
    # Weighted combination: EI gets primary weight, bonus provides exploration
    # The weight on EI should increase as we converge
    # Use a simple schedule: weight_ei = 0.7 + 0.3 * (1 - beta / 3.0)
    # When beta is high (early), weight_ei ~ 0.7; when beta is low (late), weight_ei ~ 1.0
    weight_ei = 0.7 + 0.3 * (1.0 - beta / 3.0)
    weight_ei = np.clip(weight_ei, 0.5, 1.0)
    
    scores = weight_ei * EI_norm + (1.0 - weight_ei) * bonus_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
