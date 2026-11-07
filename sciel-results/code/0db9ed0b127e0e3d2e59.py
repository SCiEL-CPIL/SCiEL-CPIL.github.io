"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a Predictive Entropy Search (PES) variant that scores each candidate
    by the expected reduction in the global minimum's posterior entropy.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the Gaussian process posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the Gaussian process posterior.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores, higher means more promising.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return -mu + 2.0 * np.maximum(sigma, 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # PES-inspired scoring:
    # For each candidate x, we want to estimate the mutual information between
    # f(x) and the location/value of the global minimum.
    # 
    # Approximation: The value at x provides information about the global min
    # if f(x) is likely to be lower than f_best. The information gain is
    # proportional to how "surprising" a low value at x would be, weighted by
    # how much it would reduce uncertainty about the global minimum.
    #
    # A practical PES approximation:
    # 1. Estimate the probability that x is near the global minimum
    #    (i.e., P(f(x) < f_best) under the GP posterior)
    # 2. Weight by the expected improvement in the global minimum
    # 3. Add an exploration bonus based on uncertainty
    
    # Probability that f(x) < f_best
    z = (f_best - mu) / sigma_safe
    prob_improve = norm.cdf(z)
    
    # Expected Improvement (EI) - how much lower could f(x) be than f_best?
    ei = (f_best - mu) * prob_improve + sigma_safe * norm.pdf(z)
    
    # The key insight for PES on multimodal functions:
    # We want to find points where discovering a new low value would
    # significantly change our belief about where the global minimum is.
    # 
    # In a multimodal landscape like Weierstrass, this means:
    # - Points with high probability of being low (high prob_improve)
    # - Points with high uncertainty (high sigma) that could reveal new basins
    # - Points that are "diverse" from already-explored regions
    #
    # Since we don't have coordinates, we approximate diversity by using
    # the posterior structure itself. Points where the GP is uncertain
    # (high sigma) but the mean suggests potential for low values are
    # most informative.
    
    # Information gain approximation:
    # The reduction in entropy about the global minimum is proportional to
    # the change in the distribution of the minimum given f(x).
    # 
    # A simpler but effective proxy: combine EI with a term that rewards
    # uncertainty in regions where the mean is already somewhat low.
    
    # Normalize EI to [0, 1]
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    # Normalize sigma to [0, 1]
    sigma_min, sigma_max = np.min(sigma_safe), np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range > 1e-12:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Combine: weight EI heavily but add exploration via sigma
    # The PES-inspired weight: points that are both likely to improve
    # and have high uncertainty are most valuable for reducing global min entropy
    
    # Use a product term that captures the interaction:
    # High EI * High Sigma = high information gain
    interaction = ei_norm * sigma_norm
    
    # Also keep a pure EI term for exploitation
    # And a pure sigma term for exploration
    
    w_ei = 0.5
    w_interaction = 0.3
    w_sigma = 0.2
    
    scores = w_ei * ei_norm + w_interaction * interaction + w_sigma * sigma_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
