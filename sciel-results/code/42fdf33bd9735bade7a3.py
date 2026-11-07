"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    
    Uses Probability of Improvement (PI) as the primary signal with a
    scaled Expected Improvement (EI) term for exploitation strength.
    PI avoids over-exploitation of deep certain valleys in Branin's
    multi-modal landscape, while the EI term provides a stronger pull
    toward candidates with high expected gain.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate point. Higher means evaluate next.
    """
    start_time = time.time()
    DEADLINE = 300.0
    SAFETY_MARGIN = 10.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    # This is robust for multi-modal landscapes as it doesn't over-weight
    # deep but certain valleys
    pi_scores = norm.cdf(z)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Scale EI to be comparable to PI (which is in [0, 1])
    # EI can be much larger than 1, so we need to scale it
    # Use a sigmoid-like scaling to map EI to [0, 1]
    # The scale parameter controls how quickly EI saturates
    # For Branin, typical EI values are in [0, 10] range
    ei_scale = 2.0
    ei_scaled = 1.0 - np.exp(-ei / ei_scale)
    
    # Combine PI and scaled EI
    # PI provides exploration-friendly ranking
    # EI provides exploitation strength
    # Weight them to balance exploration and exploitation
    
    # Adaptive weighting: as f_best approaches the optimum, favor exploitation more
    # Branin minimum is ~0.397887
    if f_best < 0.5:
        w_pi = 0.4
        w_ei = 0.6
    elif f_best < 1.0:
        w_pi = 0.5
        w_ei = 0.5
    elif f_best < 3.0:
        w_pi = 0.6
        w_ei = 0.4
    else:
        w_pi = 0.7
        w_ei = 0.3
    
    scores = w_pi * pi_scores + w_ei * ei_scaled
    
    # Add a small exploration bonus based on normalized sigma
    # This helps differentiate among highly promising candidates
    # and encourages mild exploration in uncertain regions
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        exploration_bonus = 0.05 * sigma_safe / sigma_max
    else:
        exploration_bonus = np.zeros_like(sigma_safe)
    
    scores = scores + exploration_bonus
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        # Return current scores; they should already be computed
        pass
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
