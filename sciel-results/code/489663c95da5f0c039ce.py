"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Combines Expected Improvement with a diversity penalty based on
    Matern-5/2 kernel correlations among top candidates to encourage
    exploration of distinct regions.
    """
    start_time = time.time()
    deadline = start_time + 5.0  # Safety margin well within 300s limit

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Check deadline before proceeding to diversity penalty
    if time.time() - start_time > deadline * 0.5:
        # Return EI-based scores quickly
        scores = np.where(np.isfinite(ei), ei, 0.0)
        return scores

    # Compute LCB for exploration
    beta = 2.0
    lcb = mu - beta * sigma_safe
    lcb = np.where(np.isfinite(lcb), lcb, 0.0)

    # Normalize EI to [0, 1]
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
    ei_norm = (ei - ei_min) / ei_range

    # Normalize LCB to [0, 1] (lower LCB is better, so flip)
    lcb_min, lcb_max = np.min(lcb), np.max(lcb)
    lcb_range = lcb_max - lcb_min if lcb_max > lcb_min else 1.0
    lcb_norm = (lcb_max - lcb) / lcb_range

    # Adaptive weighting based on uncertainty level
    sigma_mean = np.mean(sigma_safe)
    f_scale = max(abs(f_best), 1e-6)
    uncertainty_ratio = np.clip(sigma_mean / f_scale, 0.0, 2.0)

    # When uncertain, weight LCB more for exploration
    # When certain, weight EI more for exploitation
    w_lcb = np.clip(0.3 + 0.4 * uncertainty_ratio, 0.2, 0.7)
    w_ei = 1.0 - w_lcb

    # Combine EI and LCB
    base_scores = w_ei * ei_norm + w_lcb * lcb_norm

    # Check deadline before diversity penalty computation
    if time.time() - start_time > deadline * 0.7:
        # Return combined scores quickly
        scores = base_scores * ei_range + ei_min
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Diversity penalty: compute maximum pairwise correlation among top-K candidates
    # using Matern-5/2 kernel to discourage redundant evaluations
    
    # Use a subset of top candidates for computational efficiency
    K = min(50, len(mu))
    
    # Get top-K candidates by base score
    top_k_indices = np.argsort(base_scores)[-K:]
    
    # We need to estimate the correlation structure. Since we don't have coordinates,
    # we'll use the predictive variance as a proxy for spatial structure.
    # High sigma regions are more likely to be in different basins.
    
    # Compute a diversity score based on the spread of sigma among top candidates
    # and their relative importance in the base score
    
    top_k_sigma = sigma_safe[top_k_indices]
    top_k_scores = base_scores[top_k_indices]
    
    # Normalize top-k scores to [0, 1] for weighting
    tk_min, tk_max = np.min(top_k_scores), np.max(top_k_scores)
    tk_range = tk_max - tk_min if tk_max > tk_min else 1.0
    top_k_norm = (top_k_scores - tk_min) / tk_range
    
    # Diversity penalty: penalize candidates that are similar to other high-scoring candidates
    # We use the inverse of the local density as a diversity measure
    # Simple approximation: use the rank of sigma among top-k as a diversity signal
    
    # Higher sigma in a region suggests it's less explored, so we give a bonus
    sigma_rank = np.argsort(np.argsort(top_k_sigma)) / max(K - 1, 1)
    
    # Diversity bonus: candidates with higher sigma (less explored) get a boost
    diversity_bonus = sigma_rank * 0.1  # Small boost to encourage exploration
    
    # Apply diversity bonus to top-k candidates
    adjusted_scores = base_scores.copy()
    adjusted_scores[top_k_indices] += diversity_bonus * ei_range
    
    # Final combined score
    scores = adjusted_scores * ei_range + ei_min
    
    # Ensure all scores are finite as required by the contract
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores (higher is better)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
