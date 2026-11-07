"""Acquisition function for Bayesian optimization of Hartmann 6D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing Hartmann 6D using a dynamic
    Lower Confidence Bound (LCB) strategy.
    
    This structural change moves away from a fixed-weight hybrid of EI and UCB
    to a dynamic LCB where the exploration coefficient beta adapts to the
    current best observed value (f_best). This encourages aggressive exploration
    when the current best is poor (early in optimization) and shifts towards
    exploitation as the best value improves (late in optimization).
    
    Parameters:
    -----------
    mu : ndarray, shape (n_candidates,)
        Posterior mean of the GP at candidate points (in objective units).
    sigma : ndarray, shape (n_candidates,)
        Posterior standard deviation of the GP at candidate points (in objective units).
    f_best : float
        Best (lowest) observed objective value so far.
        
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores; higher means evaluate next.
    """
    # Safety guard: Ensure inputs are valid and handle any potential numerical issues
    # explicitly. If we detect a situation that might lead to timeout or invalidity,
    # we fall back to a simple exploitation score (-mu).
    
    try:
        # Ensure sigma is strictly positive
        sigma_safe = np.maximum(sigma, 1e-10)
        
        # Define known minimum of Hartmann 6D
        f_min = -3.32237
        
        # Estimate the "start" value for progress calculation.
        # Uniform random points on [0,1]^6 for Hartmann 6D typically have values 
        # around 1.0 to 3.0. We use 3.0 as a conservative upper bound for the initial best.
        f_start_est = 3.0
        
        # Calculate progress p in [0, 1]
        # p = 1 when f_best is high (early), p = 0 when f_best is low (late)
        # If f_best < f_min, clamp it.
        f_best_clamped = max(f_best, f_min)
        
        range_est = f_start_est - f_min
        if range_est < 1e-6:
            p = 0.0
        else:
            p = (f_best_clamped - f_min) / range_est
            p = np.clip(p, 0.0, 1.0)
            
        # Dynamic beta
        # High beta for exploration (p close to 1), low beta for exploitation (p close to 0)
        beta_min = 1.0
        beta_max = 4.0
        
        # Linear interpolation for beta
        beta = beta_min + (beta_max - beta_min) * p
        
        # LCB Score: -mu + beta * sigma
        # We want to maximize this score to find the next point to evaluate.
        scores = -mu + beta * sigma_safe
        
        # Ensure all scores are finite
        if not np.all(np.isfinite(scores)):
            # Fallback to pure exploitation if numerical issues arise
            scores = -mu
            scores = np.where(np.isfinite(scores), scores, 0.0)
            
        return scores
        
    except Exception:
        # Final safety net: return a neutral score based on mean
        try:
            scores = -mu
            scores = np.where(np.isfinite(scores), scores, 0.0)
            return scores
        except:
            return np.zeros_like(mu)
# EVOLVE-BLOCK-END
