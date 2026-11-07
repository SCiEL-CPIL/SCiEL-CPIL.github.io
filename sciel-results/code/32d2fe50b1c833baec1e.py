"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time since module load for deadline guard
_MODULE_START = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 20.0


def _time_remaining():
    elapsed = time.time() - _MODULE_START
    return _DEADLINE - elapsed - _SAFETY_MARGIN


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Posterior Sampling of Regret (PSR) combined
    with quantile-binned exploration for systematic coverage.
    
    Strategy:
    - Sample a single posterior function value vector for the 2000 candidates
      using the GP posterior (mu, sigma).
    - Use Thompson Sampling: score each candidate by its sampled value.
    - To encourage exploration of diverse regions, blend the sampled values
      with a quantile-based exploration bonus.
    
    mu, sigma: shape (2000,)
    f_best: scalar, lowest observed value
    Returns: shape (2000,) scores, higher = evaluate next
    """
    # Time guard: if we're running out of time, return a simple EI score
    if _time_remaining() < 0.5:
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    n = len(mu)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.zeros(n)
    
    # Clip sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Posterior Sampling of Regret (Thompson Sampling):
    # Sample a single function value for each candidate from N(mu, sigma^2)
    # We want to MINIMIZE, so we want low sampled values.
    # Score = -sampled_value (higher score = lower sampled value = better)
    
    # Draw a single sample from the posterior for each candidate
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(n)
    
    # The sampled values represent plausible function values.
    # We want to find the minimum, so higher score should correspond to lower sampled value.
    # score = -samples (negate so that lower sampled value gives higher score)
    
    # But we also want to incorporate exploration via quantile bins.
    # Let's compute the PSR score and blend with quantile-based exploration.
    
    # PSR score: -sampled value, normalized
    psr_raw = -samples
    psr_min = np.min(psr_raw)
    psr_max = np.max(psr_raw)
    if psr_max > psr_min + 1e-12:
        psr_norm = (psr_raw - psr_min) / (psr_max - psr_min)
    else:
        psr_norm = np.ones(n)
    
    # Quantile-based exploration: partition candidates into bins by posterior mean,
    # and within each bin, reward high uncertainty (sigma). This forces exploration
    # of diverse regions rather than clustering.
    
    # Sort by mu to create quantile bins
    order = np.argsort(mu)
    n_bins = 20
    bin_size = n // n_bins
    
    exploration_scores = np.zeros(n)
    
    for i in range(n_bins):
        bin_start = i * bin_size
        bin_end = bin_start + bin_size if i < n_bins - 1 else n
        bin_indices = order[bin_start:bin_end]
        
        # Within this bin, score based on sigma (uncertainty)
        bin_sigma = sigma_safe[bin_indices]
        
        # Normalize sigma within the bin to [0, 1]
        sigma_min = np.min(bin_sigma)
        sigma_max = np.max(bin_sigma)
        if sigma_max - sigma_min > 1e-12:
            sigma_norm = (bin_sigma - sigma_min) / (sigma_max - sigma_min)
        else:
            sigma_norm = np.ones_like(bin_sigma)
        
        exploration_scores[bin_indices] = sigma_norm
    
    # Blend PSR (exploitation via sampling) with quantile-based exploration
    # Weight PSR more heavily to ensure we converge to good regions
    alpha = 0.7  # Weight for PSR
    final_scores = alpha * psr_norm + (1 - alpha) * exploration_scores
    
    # Ensure all scores are finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_scores
# EVOLVE-BLOCK-END
