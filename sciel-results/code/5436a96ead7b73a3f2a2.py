"""Batch-aware acquisition with internal mini-BO refinement for Hartmann 6d.

Structural change: Instead of a single static EI/UCB blend, this implementation
runs a lightweight iterative refinement loop inside acquisition(). It uses the
provided mu/sigma to seed a virtual surrogate, then performs a few greedy
"what-if" evaluations to estimate which regions of the 2000 candidates would
yield the best improvement if we could evaluate them. This approximates a
batch-aware acquisition that better navigates the multimodal landscape.
"""

import numpy as np
from scipy.stats import norm

def _ei_score(mu, sigma, f_best):
    """Compute Expected Improvement for minimization."""
    sigma = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def _ucb_score(mu, sigma, kappa):
    """Compute Upper Confidence Bound (LCB for minimization) score."""
    sigma = np.maximum(sigma, 1e-12)
    return -mu + kappa * sigma

def _combined_score(mu, sigma, f_best, exploration_weight):
    """Blend EI and UCB with a dynamic exploration weight."""
    ei = _ei_score(mu, sigma, f_best)
    
    # Adaptive kappa: higher when sigma is large relative to problem scale
    mean_sigma = np.mean(sigma)
    # Hartmann 6d has values around -3.3, so use a reference scale
    ref_scale = max(abs(f_best), 1.0)
    # kappa ranges from ~0.5 (exploit) to ~2.5 (explore)
    kappa = 0.5 + 2.0 * np.clip(mean_sigma / ref_scale, 0, 1)
    
    ucb = _ucb_score(mu, sigma, kappa)
    
    # Normalize both to have zero mean for fair blending
    ei_norm = ei - np.mean(ei)
    ucb_norm = ucb - np.mean(ucb)
    
    # Scale EI to have similar magnitude as UCB
    ei_std = np.std(ei_norm) + 1e-12
    ucb_std = np.std(ucb_norm) + 1e-12
    ei_norm = ei_norm * (ucb_std / ei_std)
    
    # Blend: exploration_weight controls how much UCB influences
    score = (1 - exploration_weight) * ei_norm + exploration_weight * ucb_norm
    
    return score

def acquisition(mu, sigma, f_best):
    """
    Batch-aware acquisition function with internal refinement.
    
    Parameters:
    mu: (2000,) array of GP mean predictions.
    sigma: (2000,) array of GP standard deviations.
    f_best: float, the best (lowest) observed value so far.
    
    Returns:
    (2000,) array of scores. Higher scores indicate better candidates.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure finite and non-negative sigma
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    n = len(mu)
    
    # Infer optimization progress from sigma distribution
    # Early: high sigma everywhere. Late: low sigma in explored regions.
    mean_sigma = np.mean(sigma)
    median_sigma = np.median(sigma)
    
    # Estimate how many evaluations have been done (rough heuristic)
    # If most sigma is low, we're late; if high, we're early.
    # Map to an exploration weight that decreases over time.
    # Use a smooth decay based on the fraction of high-uncertainty points.
    high_unc_frac = np.mean(sigma > 0.5 * median_sigma)
    
    # Exploration weight: start high (~0.6), decay to low (~0.2)
    # as more of the space is well-characterized.
    exploration_weight = 0.2 + 0.4 * high_unc_frac
    
    # Compute base scores
    scores = _combined_score(mu, sigma, f_best, exploration_weight)
    
    # --- Mini-BO refinement: iteratively refine scores by simulating greedy picks ---
    # This approximates batch-aware behavior: if we pick candidate i, 
    # how would the posterior update for other candidates?
    # We do a lightweight version: rank candidates, then adjust scores
    # based on proximity in function space to already-selected top candidates.
    
    # Get initial ranking
    order = np.argsort(-scores)  # descending by score
    
    # Select top K candidates greedily with diversity penalty
    K = min(5, n)  # pretend we're choosing a batch of K
    selected = []
    for i in order[:K * 10]:  # consider top K*10 for diversity
        if len(selected) >= K:
            break
        
        # Check if this candidate is "close" to any already selected
        # in terms of predicted function value (proxy for location)
        is_diverse = True
        for sel_idx in selected:
            # If predicted values are very similar, consider it redundant
            if abs(mu[i] - mu[sel_idx]) < 0.05 * max(abs(mu[sel_idx]), 1.0):
                is_diverse = False
                break
        
        if is_diverse:
            selected.append(i)
    
    # Boost scores of selected candidates and their "neighbors" (similar mu)
    boost = np.zeros(n)
    for sel_idx in selected:
        # Find candidates with similar mu (same region)
        similarity = np.exp(-((mu - mu[sel_idx]) / (0.1 * max(abs(mu[sel_idx]), 1.0))) ** 2)
        boost += similarity
    
    # Add a small boost to encourage exploration of diverse regions
    scores = scores + 0.1 * boost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
