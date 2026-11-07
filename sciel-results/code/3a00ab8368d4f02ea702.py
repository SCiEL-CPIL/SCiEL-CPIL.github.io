"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining Expected Improvement (EI) and Lower Confidence Bound (LCB),
    normalized by evaluation cost.
    
    Parameters
    ----------
    mu : array of shape (2000,)
        GP mean predictions in objective units.
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns
    -------
    scores : array of shape (2000,)
        Acquisition scores (higher = more desirable).
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Lower Confidence Bound (LCB) - exploration term
    # kappa controls exploration; use a moderate value
    kappa = 1.5
    lcb = mu - kappa * sigma
    
    # Combine EI and LCB. 
    # EI is in objective units (improvement), LCB is also in objective units.
    # We blend them: the idea is that EI captures the probability-weighted improvement,
    # while LCB encourages exploration in uncertain regions.
    # Normalize both to be comparable.
    # Scale EI by typical magnitude
    # Use a weighted combination: 0.7 * EI + 0.3 * (-LCB) since lower LCB is better
    # Actually, for minimization, LCB = mu - kappa*sigma, lower is better.
    # So we want to maximize -LCB = -mu + kappa*sigma, or equivalently use lcb as a score where lower lcb is better.
    # Let's convert to a "desirability" score:
    # EI_desirability = ei (higher is better)
    # LCB_desirability = -lcb (higher is better, since lower lcb means more promising)
    
    # To combine, we can normalize each by its own scale
    ei_mean = np.mean(ei)
    lcb_mean = np.mean(-lcb)
    
    # Avoid division by zero
    ei_scale = max(ei_mean, 1e-9)
    lcb_scale = max(abs(lcb_mean), 1e-9)
    
    ei_norm = ei / ei_scale
    lcb_norm = (-lcb) / lcb_scale
    
    # Weighted combination
    w_ei = 0.6
    w_lcb = 0.4
    combined = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Cost-aware: divide by cost
    scores = combined / cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
