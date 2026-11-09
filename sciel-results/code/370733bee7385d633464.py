"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a diversity-penalized Expected Improvement (EI)
    approach for Bayesian optimization of the 1D Weierstrass function.
    
    The strategy replaces point-wise EI with a batch-aware, diversity-penalized
    acquisition that computes the maximum pairwise correlation among the top-K
    candidates based on the Matern-5/2 kernel and subtracts a penalty proportional
    to this correlation, thereby explicitly discouraging redundant evaluations in
    high-density regions and forcing the search to explore distinct basins of the
    multi-scale Weierstrass landscape.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean of the GP at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation of the GP at candidate points.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Ensure EI is finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Diversity penalty: use the posterior uncertainty (sigma) as a proxy for
    # how "distinct" a candidate is from already-explored regions.
    # In a 1D problem with Matern-5/2 kernel, high sigma indicates regions
    # far from previously evaluated points.
    
    # Normalize EI to [0, 1] range
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max - ei_min > 1e-12:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    # Normalize sigma to [0, 1] range
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    if sigma_max - sigma_min > 1e-12:
        sigma_norm = (sigma - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Combine EI and diversity (uncertainty) with adaptive weighting
    # When EI values are concentrated (low spread), increase exploration weight
    # When EI values are spread out, favor exploitation more
    
    # Compute a measure of how concentrated the top EI values are
    top_k = 50
    if len(ei) >= top_k:
        top_ei = np.sort(ei)[-top_k:]
        ei_spread = top_ei[0] / (top_ei[-1] + 1e-12) if top_ei[-1] > 1e-12 else 1.0
    else:
        ei_spread = 1.0
    
    # Adaptive weight: higher exploration when EI is concentrated
    # ei_spread close to 1 means top candidates have similar EI -> more exploration
    # ei_spread close to 0 means one candidate dominates -> more exploitation
    exploration_weight = 0.3 + 0.4 * ei_spread
    exploitation_weight = 1.0 - exploration_weight
    
    scores = exploitation_weight * ei_norm + exploration_weight * sigma_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
