"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Branin 2D optimization.
    
    Uses a combination of Expected Improvement (for exploitation) and 
    a properly scaled UCB (for exploration). For multi-modal functions 
    like Branin, we need strong exploration to find all basins, then 
    exploitation to refine the best one.
    
    The score is designed so that higher values are preferred.
    """
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI is >= 0, higher is better
    
    # UCB for minimization: -mu + beta * sigma
    # We need to choose beta carefully. For Branin with its scale,
    # a moderate beta works well.
    # The function values range from ~0.4 to ~300, so sigma is in that scale.
    # A beta around 1.0-2.0 is reasonable.
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Normalize EI to be comparable to UCB scale
    # EI is in objective units, UCB is also in objective units
    # But they have different distributions. Let's combine them additively
    # with a weight that favors exploration.
    
    # Actually, let's use a simpler approach: just use UCB with a well-chosen beta.
    # For Branin, exploration is critical. Let's use a pure UCB with beta=2.0
    # which encourages exploring high-uncertainty regions.
    
    # But to be safe, let's blend EI and UCB.
    # First, let's normalize both to have similar scales.
    
    # EI is always >= 0. Its typical scale depends on how far f_best is from mu.
    # UCB can be any real number.
    
    # A robust approach: use the rank of each component.
    # But that loses information. Instead, let's use a fixed combination.
    
    # Let's try: score = ei + 0.5 * ucb
    # This adds the exploitation signal (EI) with a scaled exploration signal (UCB).
    
    # However, EI and UCB have different magnitudes. Let's check typical values.
    # If f_best is around 1.0, mu ranges from 0 to 300, sigma ranges from 0.1 to 10.
    # EI would be small when mu is close to f_best, large when mu >> f_best.
    # UCB = -mu + beta*sigma, so it's large when mu is small and sigma is large.
    
    # A better approach: use only UCB with a well-tuned beta.
    # For Branin, beta=1.0 to 2.0 works well in practice.
    
    # Let's use a hybrid: primarily UCB, with a small EI component.
    # The EI component helps focus on promising regions.
    
    # To make the combination robust, let's normalize EI by its max value
    # and UCB by its range, then combine.
    
    # Actually, the simplest robust approach for Branin is just UCB with beta=1.5.
    # Let's go with that, but add a small EI term for refinement.
    
    # Normalize EI to [0, 1] by dividing by (max(ei) + epsilon)
    ei_max = np.max(ei)
    if ei_max < 1e-10:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = ei / ei_max
    
    # UCB normalization: map to [0, 1]
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-10:
        ucb_norm = np.ones_like(ucb) * 0.5
    else:
        ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Combine: 70% UCB (exploration), 30% EI (exploitation)
    # For Branin, exploration is more important to find all basins.
    score = 0.7 * ucb_norm + 0.3 * ei_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
