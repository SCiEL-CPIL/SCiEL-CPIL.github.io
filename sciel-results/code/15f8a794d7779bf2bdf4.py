# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 300s limit, 10s safety margin

def acquisition(mu, sigma, f_best):
    """
    Kriging Believer acquisition for minimization.
    
    Selects points where the posterior mean is lowest among candidates
    with sufficiently low relative uncertainty (sigma / |mu|). This is
    structurally different from EI: it directly exploits the mean prediction
    rather than computing an expected improvement integral.
    
    Parameters:
    mu: array of shape (N,)
    sigma: array of shape (N,)
    f_best: scalar, lowest observed value
    
    Returns:
    scores: array of shape (N,), higher values indicate better candidates
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return -mu
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Relative uncertainty: how uncertain are we relative to the magnitude of mu?
    # Use |f_best| as a reference scale since we're minimizing toward f_best
    ref_scale = max(abs(f_best), 1e-6)
    rel_uncertainty = sigma_safe / ref_scale
    
    # Kriging Believer: among "confident" predictions, pick the best mean.
    # We use a smooth gate: weight each point by how confident we are.
    # A point is "confident" when rel_uncertainty is small.
    # Use a sigmoid-like gate that transitions around rel_uncertainty ~ 0.1
    gate = 1.0 / (1.0 + np.exp(10.0 * (rel_uncertainty - 0.1)))
    
    # Score: negative mean (lower mu = better) weighted by confidence gate
    # Points with very high uncertainty get low scores regardless of mean
    scores = -mu * gate
    
    # For points where mu is near zero or negative (already good), 
    # the gate should let them through more aggressively
    # Add a small exploitation bias for points that beat f_best
    beat_bonus = np.where(mu < f_best, (f_best - mu) * 0.5, 0.0)
    scores += beat_bonus * gate
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
