"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 310  # 310s out of 330s budget, safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Uncertainty-Weighted Improvement (UWI) with
    a local baseline estimated from the GP posterior itself.
    
    The local baseline is a soft-min over the predicted means, weighted by
    inverse uncertainty, which approximates the best-known value in the
    neighborhood of each candidate. UWI rewards candidates where the GP
    posterior variance is high relative to the local prediction, targeting
    boundaries of explored basins.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    # Ensure inputs are clean
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Local baseline via soft-min over mu, weighted by 1/sigma ---
    # A candidate with low mu and low sigma is a "confirmed good" point.
    # We compute a soft-min: for each candidate i, the local baseline is
    # the soft-min over all j of mu_j, weighted by exp(-mu_j / tau) * (1/sigma_j).
    # This approximates the best known value in a neighborhood.
    #
    # To keep it O(n) rather than O(n^2), we use a two-pass approach:
    # First, compute a global soft-min as a baseline proxy, then modulate
    # by local uncertainty.
    
    # Soft-min parameter: controls how "sharp" the min is
    # Smaller tau -> closer to hard min; larger tau -> smoother
    tau = max(0.1 * (f_best - np.min(mu) + 1e-10), 0.01)
    
    # Weight for soft-min: prefer low mu and low sigma (confirmed good points)
    # w_j = exp(-mu_j / tau) / sigma_j
    # Use log-space for numerical stability
    log_w = -mu / tau - np.log(sigma)
    log_w_max = np.max(log_w)
    w = np.exp(log_w - log_w_max)
    w_sum = np.sum(w)
    
    if w_sum < 1e-30:
        w = np.ones(len(mu)) / len(mu)
        w_sum = 1.0
    else:
        w /= w_sum
    
    # Local baseline: weighted soft-min of mu
    # This is a single global value, but we modulate it per-candidate
    local_baseline_global = np.sum(w * mu)
    
    # Per-candidate local baseline: blend global soft-min with f_best
    # Candidates with low sigma (well-explored) should use the global soft-min
    # Candidates with high sigma (unexplored) should fall back toward f_best
    sigma_median = np.median(sigma)
    # Weight: low sigma -> trust local baseline more; high sigma -> trust f_best
    local_weight = np.exp(-sigma / (sigma_median + 1e-10))
    local_baseline = local_weight * local_baseline_global + (1 - local_weight) * f_best
    
    # --- Uncertainty-Weighted Improvement (UWI) ---
    # UWI = (local_baseline - mu) * sigma / (sigma^2 + eps)
    # This is like EI but uses local baseline and is weighted by uncertainty
    # relative to variance, which peaks at the boundary of explored basins.
    #
    # Alternative: z-scored improvement weighted by sigma
    # improvement = local_baseline - mu
    # uwi = improvement * sigma / (sigma^2 + sigma_median^2)
    # This peaks when improvement is positive and sigma is moderate.
    
    improvement = local_baseline - mu
    # UWI: reward improvement scaled by uncertainty, normalized by variance
    # This gives a bell-shaped reward in sigma: too little sigma -> no uncertainty,
    # too much sigma -> unreliable prediction.
    sigma_norm = sigma / (sigma_median + 1e-10)
    # UWI = improvement * sigma_norm / (1 + sigma_norm^2)
    # This peaks at sigma_norm = 1, i.e., sigma = sigma_median
    uwi = improvement * sigma_norm / (1.0 + sigma_norm**2)
    
    # --- LCB component for exploration of unexplored regions ---
    # Keep a strong LCB term to ensure we explore areas with high uncertainty
    # even where the local baseline isn't clearly better
    beta = 1.5
    lcb_score = -mu + beta * sigma
    
    # --- Combine UWI and LCB ---
    # UWI targets basin boundaries; LCB targets unexplored regions
    # Normalize both to comparable scales
    uwi_scale = np.std(uwi) + 1e-10
    lcb_scale = np.std(lcb_score) + 1e-10
    
    uwi_norm = uwi / uwi_scale
    lcb_norm = lcb_score / lcb_scale
    
    # Weight: primarily UWI (targets basin boundaries), with LCB for exploration
    score = 0.6 * uwi_norm + 0.4 * lcb_norm
    
    # Cost-aware normalization
    score = score / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
