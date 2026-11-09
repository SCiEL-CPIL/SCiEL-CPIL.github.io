"""Bayesian optimization acquisition for 2D RBF mixture with Matérn-5/2 GP."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = 310.0
_EPS = 1e-12


def _expected_improvement(mu, sigma, f_best):
    """Compute Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, _EPS)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _get_progress(eval_count, total_steps=25):
    """Get progress in [0, 1] over the 25 BO evaluation steps."""
    if eval_count <= 0:
        return 0.0
    return min(eval_count / float(total_steps), 1.0)


def _voronoi_ehi_score(mu, sigma, f_best, progress):
    """
    Voronoi-based Expected Hypervolume Improvement (EHVI) approximation.
    
    For each candidate, estimate the volume of the domain where it is the
    unique "winner" under posterior mean + uncertainty, weighted by the
    improvement potential over f_best.
    
    We approximate this by:
    1. Computing a "claim" value for each candidate: c_i = mu_i + beta * sigma_i
       (lower c_i = better claim for minimization)
    2. Using a soft-min over all candidates to estimate each one's Voronoi cell
       volume via softmax attention (higher attention = larger claimed region)
    3. Weighting by the expected improvement potential: max(0, f_best - c_i)
    
    This directly optimizes probability of discovering new basins in the
    multimodal RBF mixture.
    """
    sigma_safe = np.maximum(sigma, _EPS)
    n = mu.size
    
    # Beta controls exploration: larger early, smaller late
    # Early: beta ~ 2.5 (aggressive exploration of uncertain regions)
    # Late:  beta ~ 0.3 (exploit known good regions)
    beta = 2.5 - 2.2 * (1.0 - np.exp(-5.0 * progress))
    beta = np.clip(beta, 0.2, 3.0)
    
    # Claim value: lower is better
    claim = mu + beta * sigma_safe
    
    # Improvement potential: how much better than f_best could this be?
    # Use soft version: max(0, f_best - claim) approximated smoothly
    improvement_potential = np.maximum(f_best - claim, 0.0)
    
    # Estimate Voronoi cell size via softmax attention.
    # For candidate i, its "cell" is the region where it has the lowest claim.
    # We approximate cell volume by: exp(-claim_i / T) / sum_j exp(-claim_j / T)
    # Lower claim -> higher weight -> larger estimated cell
    # Use temperature T that scales with sigma to handle varying uncertainty
    T = np.mean(sigma_safe) + _EPS
    T = max(T, _EPS)
    
    # Softmax weights (stable computation)
    neg_claim_over_T = -claim / T
    max_neg = np.max(neg_claim_over_T)
    exp_vals = np.exp(neg_claim_over_T - max_neg)
    sum_exp = np.sum(exp_vals)
    if sum_exp < _EPS:
        weights = np.ones(n) / n
    else:
        weights = exp_vals / sum_exp
    
    # Voronoi EHI score: cell_weight * improvement_potential
    # This rewards candidates that (a) claim a large region AND (b) have high
    # improvement potential in that region
    score = weights * improvement_potential
    
    # Add a small pure-exploration term to ensure diversity in early stages
    # This prevents all weight collapsing to one point
    explore_term = sigma_safe / (np.max(sigma_safe) + _EPS)
    explore_weight = 0.15 * (1.0 - progress)  # 0.15 early, 0 late
    score = score + explore_weight * explore_term
    
    return score


def _soft_ei_scale(ei, top_k=50):
    """
    Compute a smooth scale factor for EI based on top-k EI values.
    Uses log-sum-exp style normalization for stability.
    Returns a scalar scale such that top EI values are O(1).
    """
    if ei.size == 0:
        return 1.0
    # Get top-k EI values
    k = min(top_k, ei.size)
    top_vals = np.partition(ei, -k)[-k:]
    top_vals = top_vals[top_vals > _EPS]
    if top_vals.size == 0:
        return 1.0
    # Use the max of top-k as a robust scale
    scale = np.max(top_vals)
    return max(scale, _EPS)


# Track evaluation count
_eval_count = 0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of 2D RBF mixture.
    
    Uses a Kriging-Believer quantile-based score blended with a
    properly-scaled Expected Improvement term. The KB score smoothly
    transitions from aggressive exploration (low quantile) to
    exploitation (mean) as progress increases.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior standard deviation
        f_best: float - lowest observed value
    
    Returns:
        array of shape (2000,) - scores, higher means evaluate next
    """
    global _eval_count
    
    # Deadline guard: fall back to pure EI
    if time.time() - _T0 > _DEADLINE:
        sigma_safe = np.maximum(sigma, _EPS)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0)
    
    # Increment evaluation counter
    _eval_count += 1
    
    sigma_safe = np.maximum(sigma, _EPS)
    progress = _get_progress(_eval_count)
    
    # Voronoi EHI score (primary signal)
    voronoi_score = _voronoi_ehi_score(mu, sigma_safe, f_best, progress)
    
    # Compute EI (secondary exploitation anchor)
    ei = _expected_improvement(mu, sigma_safe, f_best)
    
    # Scale EI using top-k based scaling for stability
    ei_scale = _soft_ei_scale(ei, top_k=50)
    ei_scaled = ei / ei_scale
    
    # Blend weights: Voronoi EHI dominates, EI provides a small exploitation anchor
    # w_ei increases with progress to anchor near the incumbent later
    w_ei = 0.03 + 0.35 * progress  # 0.03 -> 0.38
    w_vor = 1.0 - w_ei
    
    # Normalize Voronoi score to be O(1) by its own spread
    vor_range = np.max(voronoi_score) - np.min(voronoi_score)
    if vor_range > _EPS:
        vor_norm = (voronoi_score - np.min(voronoi_score)) / vor_range
    else:
        vor_norm = np.zeros_like(voronoi_score)
    
    # Blend
    score = w_vor * vor_norm + w_ei * ei_scaled
    
    # Apply a soft-max sharpening to emphasize top candidates
    # This helps when the Voronoi score has a clear winner
    score_max = np.max(score)
    score_min = np.min(score)
    if score_max - score_min > _EPS:
        # Sharpen: amplify differences
        sharpened = score - score_min
        sharpened = sharpened / (sharpened.max() + _EPS)
        # Blend original and sharpened
        score = 0.7 * score + 0.3 * sharpened
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
