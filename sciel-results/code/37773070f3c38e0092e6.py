import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with an
    adaptive exploration bonus based on relative uncertainty.
    
    The exploration bonus rewards candidates whose sigma is above the
    median sigma of the candidate set, encouraging the search to visit
    under-explored regions. This is a structural change from pure EI:
    it decouples the exploration incentive from the improvement
    likelihood, so that high-uncertainty points get a boost even when
    their mean is not yet promising.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against degenerate inputs
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: reward points with above-median uncertainty
    # Normalize sigma to [0, 1] range based on current candidate set
    s_min = np.min(sigma_safe)
    s_max = np.max(sigma_safe)
    s_range = s_max - s_min
    if s_range < 1e-9:
        # All sigmas are essentially equal; no exploration signal
        return ei
    
    # Relative uncertainty: 0 at min sigma, 1 at max sigma
    rel_sigma = (sigma_safe - s_min) / s_range
    
    # Use a nonlinear weighting: emphasize the top half of uncertainty
    # This makes the bonus stronger for truly uncertain points
    exploration_bonus = 0.15 * np.maximum(rel_sigma - 0.5, 0.0) * 2.0
    
    # Combine: EI dominates, but exploration bonus provides a structural
    # push toward under-explored regions
    scores = ei + exploration_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
