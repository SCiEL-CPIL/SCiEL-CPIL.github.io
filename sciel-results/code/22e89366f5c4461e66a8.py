"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling-inspired acquisition function for Bayesian optimization
    of Ackley 2D.
    
    The score is -mu + beta * sigma, where beta is a dynamic exploration weight
    that scales with the current best value. This encourages aggressive
    exploration early (when f_best is large) and sharp focus on exploitation
    as f_best approaches the minimum (0).
    
    To make the exploration weight robust to the scale of sigma, we normalize
    sigma by its quantiles to create a bounded exploration term that adapts
    to the current posterior uncertainty distribution.
    
    mu: (N,) posterior mean
    sigma: (N,) posterior std
    f_best: best observed value so far
    
    Returns: (N,) scores, higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and clip tiny values
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Quantile-normalize sigma to [0, 1] range to make exploration weight
    # scale-invariant and bounded
    q10 = np.quantile(sigma_safe, 0.10)
    q90 = np.quantile(sigma_safe, 0.90)
    q_range = max(q90 - q10, 1e-8)
    sigma_norm = (sigma_safe - q10) / q_range
    sigma_norm = np.clip(sigma_norm, 0.0, 1.0)
    
    # Dynamic exploration weight:
    # When f_best is large (far from minimum), explore more aggressively.
    # When f_best approaches 0, reduce exploration and focus on exploitation.
    # f_best for Ackley 2D ranges from ~0 to ~20+.
    # Use a smooth transition: beta_high early, beta_low late.
    f_abs = abs(f_best)
    # beta transitions from 2.0 (when f_best ~ 20) to 0.5 (when f_best ~ 0)
    # Using a hyperbolic tangent for smooth transition
    beta_high = 2.0
    beta_low = 0.5
    transition_center = 5.0  # f_best value where transition is centered
    transition_width = 3.0   # width of transition
    beta = beta_low + (beta_high - beta_low) * (
        0.5 * (1.0 + np.tanh((transition_center - f_abs) / transition_width))
    )
    
    # Score: negative mean plus exploration term
    score = -mu + beta * sigma_norm
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
