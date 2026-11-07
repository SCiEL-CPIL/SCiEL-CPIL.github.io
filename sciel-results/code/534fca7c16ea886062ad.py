"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidates.
    
    Uses a hybrid strategy:
    - Primary: Expected Improvement (EI) for exploitation
    - Secondary: Quantile-based UCB for exploration to prevent clustering
    
    The function returns 2000 finite scores, higher meaning evaluate next.
    """
    # Track elapsed time for deadline guard
    t_start = time.time()
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    
    # Compute z-scores for EI
    z = (f_best - mu) / sigma
    
    # Handle numerical stability for large negative z
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Quantile-based exploration: divide candidates into 20 quantile bins of mu
    # and select the best UCB point from each bin
    n_candidates = len(mu)
    n_bins = 20
    
    # Compute quantile edges for mu
    quantile_edges = np.quantile(mu, np.linspace(0, 1, n_bins + 1))
    
    # Assign each candidate to a bin based on its mu value
    bin_indices = np.digitize(mu, quantile_edges[1:-1])
    
    # UCB score: mu - beta * sigma (minimization, so lower mu is better, 
    # but we want to explore high uncertainty, so we use -mu + beta*sigma)
    # For minimization: we want low mu and high sigma
    # UCB for minimization: score = -mu + beta * sigma (higher is better for exploration)
    beta = 1.0
    ucb = -mu + beta * sigma
    
    # For each bin, find the candidate with the highest UCB score
    best_ucb_per_bin = np.full(n_bins, -np.inf)
    best_ucb_idx_per_bin = np.full(n_bins, -1, dtype=int)
    
    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            bin_ucb = ucb[mask]
            best_in_bin = np.argmax(bin_ucb)
            global_idx = np.where(mask)[0][best_in_bin]
            best_ucb_per_bin[b] = bin_ucb[best_in_bin]
            best_ucb_idx_per_bin[b] = global_idx
    
    # Blend EI and quantile-UCB
    # The idea: use EI as the primary score, but boost candidates that are 
    # the best UCB in their quantile bin to ensure diversity
    
    # Normalize EI to [0, 1] for blending
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max > ei_min:
        ei_normalized = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_normalized = np.zeros_like(ei)
    
    # Create a diversity bonus: candidates that are best UCB in their bin get a boost
    diversity_bonus = np.zeros(n_candidates)
    for b in range(n_bins):
        if best_ucb_idx_per_bin[b] >= 0:
            diversity_bonus[best_ucb_idx_per_bin[b]] = 1.0
    
    # Final score: blend of normalized EI and diversity bonus
    # Weight EI heavily, but give a meaningful boost to diverse exploratory points
    w_ei = 0.8
    w_div = 0.2
    scores = w_ei * ei_normalized + w_div * diversity_bonus
    
    # Scale back to EI scale for consistency
    if ei_max > ei_min:
        scores = scores * (ei_max - ei_min) + ei_min
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if we're taking too long, return a simpler score
    t_elapsed = time.time() - t_start
    if t_elapsed > 250:  # Safety margin before 300s limit
        # Fall back to pure EI
        return ei
    
    return scores
# EVOLVE-BLOCK-END
