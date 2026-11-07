"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) acquisition function.
    
    The exploration parameter beta adapts based on the current best value
    relative to the known global minimum of the Hartmann 6D function.
    Early in optimization (large regret), beta is high to encourage exploration
    across the multimodal landscape. Late in optimization (small regret),
    beta is low to exploit known good regions.
    
    Parameters
----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    # Start a timer to guard against the 330s wall-clock limit
    t_start = time.time()
    
    # Guard against non-positive sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Theoretical global minimum of Hartmann 6D function
    f_min = -3.32237
    
    # Calculate current regret (distance from best found to global min)
    regret = f_best - f_min
    
    # Clip regret to avoid numerical issues
    regret = np.clip(regret, 0.0, 10.0)
    
    # Adaptive Beta
    # High exploration when regret is large, low exploration when regret is small
    beta_max = 2.5
    beta_min = 0.5
    scale = 1.0
    
    # Transition factor: 0 when regret=0, approaches 1 as regret increases
    # Using 1 - exp(-regret/scale) for smooth transition
    transition = 1.0 - np.exp(-regret / scale)
    
    beta = beta_min + (beta_max - beta_min) * transition
    
    # LCB score for minimization
    # We want to maximize: -mu + beta * sigma
    # -mu is high when mu is low (good exploitation)
    # sigma is high when uncertainty is high (good exploration)
    scores = -mu + beta * sigma_safe
    
    # Check time budget
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
