"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """Hybrid POI + UCB acquisition for Weierstrass minimization.
    
    Combines:
    1. Probability of Improvement (POI): P(f(x) < f_best) - smooth exploitation
    2. Uncertainty bonus: sigma-based exploration term scaled by f_best
    
    This balances finding new global minima (exploration in high-sigma regions)
    with refining known good regions (exploitation where mu is low).
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Normalize f_best to get a reasonable scale for exploration
    # If f_best is very small, reduce exploration; if larger, increase it
    f_scale = np.abs(f_best) + 1e-6
    
    # Probability of Improvement: P(mu - sigma*z < f_best) = Phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # UCB term: explore high uncertainty regions
    # Scale exploration by current best value to adapt to problem scale
    beta = 1.5 * f_scale
    ucb_term = beta * sigma_safe
    
    # Combine: weighted sum of POI (exploitation) and UCB (exploration)
    # POI is in [0,1], ucb_term can be large; normalize ucb by typical scale
    # Use a soft combination that emphasizes exploration early, exploitation later
    
    # Adaptive weighting: when f_best is still far from 0, explore more
    # When f_best is close to 0, exploit more
    exploit_weight = 1.0 / (1.0 + f_scale)
    explore_weight = 1.0 - exploit_weight
    
    # Normalize UCB term to be comparable to POI
    # Typical sigma scale is around 0.01-0.1, f_scale similar
    ucb_normalized = ucb_term / (f_scale + 1e-6)
    
    score = exploit_weight * poi + explore_weight * ucb_normalized
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
