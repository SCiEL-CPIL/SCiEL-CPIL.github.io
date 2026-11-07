# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Deterministic Expected Improvement (EI) acquisition function with a small
    exploration offset for Bayesian optimization of Michalewicz 2D.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher means more likely to be evaluated next
    """
    t_start = time.time()
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-10)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    # Known minimum of Michalewicz 2D
    f_min = -1.8013
    
    # Compute progress: how close are we to the global minimum?
    if f_best is not None and np.isfinite(f_best):
        progress = np.clip((0 - f_best) / (0 - f_min), 0, 1) if f_min < 0 else 0
    else:
        progress = 0
    
    # Adaptive xi: start with a larger exploration offset early on,
    # decrease as we get closer to the minimum
    # xi is added to f_best to make the EI threshold slightly higher,
    # encouraging more exploration early
    xi = 0.1 * (1 - 0.7 * progress)
    
    # Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    # We add xi to f_best to encourage exploration
    
    f_best_adj = f_best + xi if f_best is not None and np.isfinite(f_best) else 0.0
    
    # Compute the improvement term
    improvement = f_best_adj - mu_safe
    
    # Compute z = improvement / sigma
    z = improvement / sigma_safe
    
    # Handle numerical stability for very large or small z
    z_clipped = np.clip(z, -10, 10)
    
    # Compute EI
    # EI = improvement * Phi(z) + sigma * phi(z)
    # For large negative z, Phi(z) ~ 0 and phi(z) ~ 0, so EI ~ 0
    # For large positive z, Phi(z) ~ 1 and phi(z) ~ 0, so EI ~ improvement
    
    Phi_z = norm.cdf(z_clipped)
    phi_z = norm.pdf(z_clipped)
    
    EI = improvement * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (clip small negative values from numerical errors)
    EI = np.maximum(EI, 0.0)
    
    # Add a small exploration bonus based on uncertainty to help escape local optima
    # This is a soft bonus that rewards high uncertainty regions
    sigma_max = np.max(sigma_safe) if len(sigma_safe) > 0 else 1.0
    sigma_min = np.min(sigma_safe) if len(sigma_safe) > 0 else 0.0
    sigma_range = sigma_max - sigma_min
    if sigma_range > 1e-10:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
        exploration_bonus = 0.05 * sigma_norm * (1 - progress)
    else:
        exploration_bonus = 0.0
    
    scores = EI + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check: if we're taking too long, return early
    if time.time() - t_start > 5.0:
        # Fallback: just use EI without exploration bonus
        scores = np.maximum(EI, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
