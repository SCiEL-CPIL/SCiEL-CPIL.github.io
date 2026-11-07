"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of 1D sphere on [-5, 5].
    
    Strategy: Aggressive Exploitation.
    The sphere function x^2 is unimodal and smooth. The primary risk in BO
    for such functions is not missing the global minimum (which is unique
    and central) but rather wasting evaluations exploring regions that are
    clearly suboptimal. 
    
    The current hybrid approach (EI + UCB blend) tends to explore too much
    early on. By using a pure Expected Improvement (EI) strategy, we focus
    strictly on points that have a high probability of improving upon the
    current best. 
    
    To ensure we don't get stuck in local minima (though unlikely for x^2)
    or suffer from numerical instability where sigma is tiny, we add a very
    small floor to sigma. 
    
    EI for minimization:
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    This function is maximized at the point where the posterior mean is
    lowest relative to the uncertainty. For a unimodal function, this
    effectively drives the search towards the peak of the posterior
    probability of being the minimum.
    """
    # Ensure sigma is positive and finite to avoid division by zero or inf
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # norm.cdf is Phi, norm.pdf is phi
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    # If mu is very high, ei is 0. If mu is very low, ei is high.
    # This is robust.
    score = np.where(np.isfinite(ei), ei, 0.0)
    
    return score
# EVOLVE-BLOCK-END
