"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute constants for speed
_EPS = 1e-9


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a hybrid acquisition that blends UCB (exploration) and EI (exploitation)
    with an adaptive weight that shifts from exploration-heavy to exploitation-heavy
    as f_best approaches 0.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points (in objective units)
    sigma : ndarray, shape (2000,)
        GP posterior std at candidate points (in objective units)
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure finite values
    sigma_safe = np.maximum(sigma, _EPS)
    
    # Compute z for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) - for minimization, we want low mu + exploration
    # UCB for minimization: mu - beta * sigma (we want to minimize, so subtract)
    # But for acquisition score (higher = better), we use: -mu + beta * sigma
    # Actually, let's think about this more carefully.
    # For minimization:
    #   EI: high when f_best - mu is large and sigma is moderate
    #   UCB: we want to explore where sigma is high, so score = -mu + beta * sigma
    #   PI (Probability of Improvement): norm.cdf(z)
    
    # Adaptive beta: start high (exploration) and decrease as f_best -> 0
    # f_best for Ackley 2D minimum is 0, so progress = 1 - f_best (clipped)
    # When f_best is large (early), we want more exploration
    # When f_best is small (late), we want more exploitation
    
    # Use a smooth transition
    # progress: 0 when f_best is large, 1 when f_best approaches 0
    # Ackley 2D max value is around 30, min is 0
    # Let's use: progress = clip(1 - f_best / 10, 0, 1) as a rough measure
    # But f_best can be negative in some cases? No, Ackley 2D minimum is 0.
    
    # Better: use the magnitude of f_best relative to typical scale
    # Early on f_best might be ~10-20, later ~0.1-1
    # Let's use: exploration_weight = exp(-f_best / 5) * 0.8 + 0.2
    # This gives ~1.0 early (f_best large) and ~0.2 late (f_best small)
    
    # Actually, let's use a simpler approach:
    # beta starts at 2.0 and decreases to 0.5 as f_best -> 0
    # progress = 1 - min(f_best / 20.0, 1.0)  # 0 early, 1 late
    # beta = 2.0 - 1.5 * progress
    
    # Let's use:
    f_best_clipped = max(f_best, 0.0)
    # Scale: assume initial f_best ~ 10-20, final ~ 0
    # progress = 1 - f_best / 15 (clipped to [0, 1])
    progress = 1.0 - min(f_best_clipped / 15.0, 1.0)
    progress = max(0.0, min(1.0, progress))
    
    # Beta for UCB: high early, low late
    beta = 2.5 - 2.0 * progress  # ranges from 2.5 to 0.5
    
    # UCB score (higher = more promising for minimization)
    ucb = -mu + beta * sigma_safe
    
    # Blend EI and UCB
    # Weight: early -> more UCB, late -> more EI
    # w_ei = progress, w_ucb = 1 - progress
    # But we need to normalize both to comparable scales
    
    # Normalize EI and UCB to have similar scales
    # EI is in objective units, UCB is also in objective units (roughly)
    # Let's use a weighted sum with adaptive weights
    
    w_ei = 0.3 + 0.7 * progress  # 0.3 early, 1.0 late
    w_ucb = 1.0 - w_ei * 0.5     # 0.85 early, 0.5 late
    
    # Normalize to have zero mean and unit std for blending
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + _EPS
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + _EPS
    
    ei_norm = (ei - ei_mean) / ei_std
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Combine
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Add a small epsilon to ensure finite and non-negative scores
    # Actually, we just need finite scores, higher = better
    # Ensure no NaN or Inf
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Should not happen, but return what we have
        pass
    
    return scores
# EVOLVE-BLOCK-END
