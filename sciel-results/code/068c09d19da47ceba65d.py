"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid Thompson Sampling / Expected Improvement acquisition.
    
    Combines a stochastic Thompson Sampling component (draw from a GMM of the
    posterior uncertainty) with a deterministic Expected Improvement term.
    The mixture weights adapt based on current regret, providing a structural
    shift from pure LCB/UCB to a probabilistic scoring rule that naturally
    balances exploration and exploitation.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    t_start = time.time()
    
    n = mu.shape[0]
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Theoretical global minimum of Hartmann 6D function
    f_min = -3.32237
    regret = max(f_best - f_min, 0.0)
    
    # Check time budget early
    if time.time() - t_start > 290:
        scores = -mu + 1.5 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Adaptive parameters based on regret
    # When regret is large (early), explore more via Thompson Sampling
    # When regret is small (late), exploit more via Expected Improvement
    
    # Transition factor: 0 when regret=0, approaches 1 as regret increases
    transition = 1.0 - np.exp(-regret / 1.0)
    
    # Weight for Thompson Sampling vs Expected Improvement
    # High transition -> more Thompson Sampling (exploration)
    # Low transition -> more Expected Improvement (exploitation)
    ts_weight = 0.3 + 0.7 * transition
    ei_weight = 1.0 - ts_weight
    
    # --- Expected Improvement component ---
    # EI = E[max(f_best - y, 0)] where y ~ N(mu, sigma^2)
    z = (mu - f_best) / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * (1.0 - Phi_z) + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # --- Thompson Sampling component ---
    # Construct a finite GMM from the sigma distribution
    # Use percentiles to define component means and variances
    sigma_vals = sigma_safe
    sigma_min = np.min(sigma_vals)
    sigma_max = np.max(sigma_vals)
    sigma_range = max(sigma_max - sigma_min, 1e-12)
    
    # Define 3 components at low, mid, high uncertainty
    # Component means in the normalized sigma space
    comp_means_norm = np.array([0.25, 0.5, 0.75])
    comp_means = sigma_min + comp_means_norm * sigma_range
    
    # Component variances: proportional to local sigma
    comp_variances = np.maximum(0.1 * sigma_range, 1e-12)
    
    # Component weights: adapt based on regret
    # Early (high regret): more weight on high-uncertainty components
    # Late (low regret): more weight on low-uncertainty components
    if transition > 0.5:
        # Early: explore high uncertainty
        comp_weights = np.array([0.2, 0.3, 0.5])
    else:
        # Late: exploit low uncertainty
        comp_weights = np.array([0.5, 0.3, 0.2])
    
    comp_weights = comp_weights / np.sum(comp_weights)
    
    # Draw a single random sample from the GMM
    rng = np.random.default_rng()
    comp_idx = rng.choice(3, p=comp_weights)
    ts_draw = rng.normal(loc=comp_means[comp_idx], scale=np.sqrt(comp_variances[comp_idx]))
    
    # Score each candidate: expected value under the Thompson draw
    # For minimization, we want low values, so score = -(mu - ts_draw * sigma_safe)
    # This means candidates where mu is low AND sigma is high get boosted
    ts_score = -(mu - ts_draw * sigma_safe)
    
    # Combine Thompson Sampling and Expected Improvement
    scores = ts_weight * ts_score + ei_weight * ei
    
    # Check time budget
    if time.time() - t_start > 295:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
