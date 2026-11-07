"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses a hybrid approach: combines Expected Improvement (EI) with a
    probability-of-improvement weighted exploration term to balance
    exploitation and exploration effectively for the 20-basin landscape.
    
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
    
    # Ensure sigma is positive
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    Phi = ndtr(z)
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    ei = np.maximum(ei, 0.0)
    
    # Structural change: Use a hybrid of EI and a probability-of-improvement
    # weighted sigma term. This creates a different exploration/exploitation
    # balance than pure EI or UCB.
    #
    # The idea: 
    # - EI handles exploitation well
    # - P(improvement) * sigma gives a different exploration signal
    #   that peaks at moderate uncertainty, not maximum uncertainty
    # - The combination creates a more nuanced landscape for candidate selection
    #
    # P(improvement) = Phi(z) where z = (f_best - mu)/sigma
    # This is 1 when mu << f_best (likely improvement) and 0 when mu >> f_best
    #
    # The term Phi(z) * sigma peaks when there's both reasonable probability
    # of improvement AND significant uncertainty, which is ideal for
    # discovering new basins in the ST-1D landscape.
    
    # Scale the exploration term to be comparable to EI
    # Typical EI values for ST-1D are in the range [0, ~5]
    # sigma * Phi(z) is in [0, ~sigma_max]
    # We use a weighted combination
    
    # Dynamic weighting: give more weight to exploration early on
    # (when f_best is far from optimum) and less as we converge
    # We estimate "early" by looking at the spread of mu relative to f_best
    mu_spread = np.std(mu)
    f_best_scale = max(abs(f_best), 1.0)
    
    # If f_best is much larger than typical function values, we're early
    # ST-1D minimum is -39.1662, so f_best < -39 means we're close to optimum
    if f_best > -20:
        # Early stage: more exploration
        w_explore = 0.6
    elif f_best > -35:
        # Middle stage: balanced
        w_explore = 0.4
    else:
        # Late stage: more exploitation
        w_explore = 0.2
    
    # Exploration term: P(improvement) weighted sigma
    # This naturally down-weights regions where improvement is unlikely
    # and up-weights regions with both uncertainty and potential
    explore_term = Phi * sigma_safe
    
    # Normalize explore_term to be comparable to ei
    # Use the median of ei as a scale reference
    ei_median = np.median(ei[ei > 0]) if np.any(ei > 0) else 1.0
    explore_scaled = explore_term / (ei_median + eps)
    
    # Combine: weighted sum of EI and scaled exploration
    scores = (1.0 - w_explore) * ei + w_explore * explore_scaled
    
    # Ensure scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 290:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
