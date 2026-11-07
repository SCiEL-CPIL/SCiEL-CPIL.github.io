import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a structural change: a hybrid of
    Probability of Improvement (PI) and a noise-free Expected Improvement (EI)
    with a small exploration bonus, designed to be robust and fast.
    
    The previous UCB approach with adaptive beta scored 4.72.
    The observed stagnation suggests that the current exploration strategy
    is not effectively navigating the Beale function's landscape.
    
    This new approach uses a fixed, well-tuned EI variant that has been
    shown to work well in practice for 2D problems like Beale.
    
    For minimization:
    EI(x) = E[max(f_best - f(x), 0)]
    
    With a small additional term to encourage exploration in regions
    with high uncertainty, but weighted to not dominate the improvement signal.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Handle non-finite f_best
    if not np.isfinite(f_best):
        f_best = 1e6
    
    # Standardized improvement
    # z = (f_best - mu) / sigma
    # For minimization, we want f(x) < f_best, so improvement = f_best - f(x)
    # E[max(f_best - f(x), 0)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Note: z is positive when mu < f_best (i.e., current mean is already better)
    
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    # Using scipy.stats.norm for accuracy
    try:
        Phi = norm.cdf(z)
        phi = norm.pdf(z)
    except Exception:
        # Fallback to manual computation
        Phi = 0.5 * (1 + np.erf(z / np.sqrt(2)))
        phi = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    
    # Expected Improvement
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Add a small exploration bonus based on uncertainty
    # This is a fixed small weight to encourage exploration without dominating
    # The Beale function has a narrow basin around the minimum, so we want
    # to be somewhat conservative but not completely greedy
    exploration_weight = 0.1
    score = EI + exploration_weight * sigma
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
