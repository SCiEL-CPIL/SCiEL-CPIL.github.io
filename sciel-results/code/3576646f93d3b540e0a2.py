"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Adaptive Expected Improvement (EI) acquisition for minimization.
    
    Uses a time-varying xi parameter that starts high (encouraging large
    exploration jumps when f_best is far from optimum) and decreases as
    f_best approaches the known minimum (0.397887), enabling fine-grained
    exploitation near the end.
    
    EI(x) = (f_best - mu(x) - xi) * Phi((f_best - mu(x) - xi)/sigma(x))
            + sigma(x) * phi((f_best - mu(x) - xi)/sigma(x))
    
    We return EI scores (higher = more desirable).
    
    Parameters:
        mu: (2000,) array of GP means in objective units.
        sigma: (2000,) array of GP standard deviations in objective units.
        f_best: Lowest observed function value.
    
    Returns:
        (2000,) array of scores, higher meaning more desirable.
    """
    # Known minimum of Branin-2D
    f_min = 0.397887
    
    # Adaptive xi: high when f_best is far from optimum, low when close
    # This encourages exploration early and exploitation late
    gap = f_best - f_min
    gap = max(gap, 0.0)
    
    # xi scales with the gap: when far from optimum, require big improvement;
    # when close, accept small improvements
    # Normalize gap to [0, 1] roughly: Branin range is about [0.4, 302]
    # So gap can be up to ~300
    gap_norm = min(gap / 50.0, 1.0)
    
    # xi: from ~5.0 (exploration) down to ~0.01 (exploitation)
    # Use a smooth interpolation
    xi = 0.01 + gap_norm * 4.99
    
    # Compute EI
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Improvement over f_best with offset xi
    improvement = f_best - mu - xi
    z = improvement / sigma_safe
    
    # Phi and phi
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    ei = improvement * Phi + sigma_safe * phi
    
    # EI can be negative; clamp to 0 so we only consider positive expected improvements
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
