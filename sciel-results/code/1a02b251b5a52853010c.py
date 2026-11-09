"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    Uses Hypervolume-based acquisition (approximated EHI).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the Gaussian process posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the Gaussian process posterior.
    f_best : float
        The best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher scores indicate more promising candidates.
    """
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Check time budget
    if time.time() > _DEADLINE - _MARGIN:
        # Fallback to simple EI if running out of time
        z = (f_best - mu) / sigma
        from scipy.stats import norm
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei

    # Hypervolume-based acquisition
    # We approximate the expected hypervolume improvement.
    # For a single objective minimization, the hypervolume improvement of a point x
    # over the current best f_best is related to how much better x is than f_best.
    # However, EHI is more complex. A common approximation is to integrate the 
    # probability that f(x) < t over a range of t, weighted by the "improvement" over f_best.
    
    # A simpler and effective approach: 
    # The expected hypervolume improvement (EHI) for a single objective can be approximated
    # by integrating the probability that the function value is below a threshold t,
    # for t in [f_best - delta, f_best], where delta is a scale factor.
    
    # Let's use a discretized integration over objective values.
    # The hypervolume dominated by a point y (relative to a reference point) is (ref - y) if y < ref.
    # Here, the "reference" is effectively f_best, but we want to measure improvement.
    # Actually, standard EHI for minimization with a single reference point r (where r > f_best) is:
    # EHI = E[max(0, r - f(x))] - (r - f_best)
    # This simplifies to E[max(0, f_best - f(x))] if we set r = f_best? No, that's EI.
    
    # Let's stick to a robust approximation of EHI that differs from EI.
    # EHI can be computed by integrating the CDF of the improvement.
    # For a normal distribution N(mu, sigma^2), the expected improvement is:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Hypervolume improvement is slightly different. It considers the area under the 
    # probability distribution curve that is "better" than f_best.
    # For a single objective, EHI with reference point r is:
    # EHI = (r - mu) * Phi(z) + sigma * phi(z) - (r - f_best)
    # where z = (f_best - mu) / sigma.
    # If we set r = f_best, EHI = EI.
    
    # To make it materially different, we can use a reference point r that is slightly 
    # worse than f_best, or use a different weighting.
    # However, a common variant is to use the "Expected Hypervolume Improvement" 
    # with a reference point that is the worst observed value or a fixed offset.
    
    # Let's try a different approach: 
    # Use the probability of improvement (POI) combined with the expected improvement.
    # Or, use a discretized EHI.
    
    # Let's implement a discretized EHI with a reference point r = f_best + offset.
    # The offset can be scaled by the current uncertainty or a fixed value.
    # Let's use r = f_best + 1.0 (arbitrary but reasonable for Branin scale).
    
    r = f_best + 1.0
    z = (f_best - mu) / sigma
    
    from scipy.stats import norm
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # EHI = (r - mu) * Phi(z) + sigma * phi(z) - (r - f_best)
    ehi = (r - mu) * Phi + sigma * phi - (r - f_best)
    
    # Ensure non-negative
    ehi = np.maximum(ehi, 0.0)
    
    # Ensure finite
    ehi = np.where(np.isfinite(ehi), ehi, 0.0)
    
    return ehi
# EVOLVE-BLOCK-END
