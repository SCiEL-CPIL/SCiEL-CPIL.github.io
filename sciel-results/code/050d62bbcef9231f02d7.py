import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using adaptive UCB with quantile-based normalization.
    
    Core: score = -mu + beta * sigma, where beta = f_best / med_sigma (adaptive).
    Refinement: rank-based normalization to ensure robust scale invariance,
    with weight shifted toward exploration when f_best >> median(mu).
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = len(mu)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Core adaptive beta: scale exploration by potential improvement relative to uncertainty
    med_sigma = np.median(sigma)
    if med_sigma < 1e-6:
        med_sigma = 1e-6
    
    beta = max(f_best / med_sigma, 0.5)
    beta = min(beta, 5.0)
    
    # Core UCB score for minimization: maximize -mu + beta * sigma
    ucb_score = -mu + beta * sigma
    
    # Quantile-based normalization: rank mu and sigma independently
    # This provides robust scale invariance
    mu_ranks = np.argsort(np.argsort(mu)).astype(np.float64) / max(n - 1, 1)  # [0, 1]
    sigma_ranks = np.argsort(np.argsort(sigma)).astype(np.float64) / max(n - 1, 1)  # [0, 1]
    
    # Normalized exploitation: lower mu rank is better -> exploit_score = 1 - mu_rank
    exploit_norm = 1.0 - mu_ranks
    # Normalized exploration: higher sigma rank is better -> explore_norm = sigma_rank
    explore_norm = sigma_ranks
    
    # Dynamic weight: shift toward exploration when f_best >> median(mu)
    med_mu = np.median(mu)
    # gap_measure: how much f_best exceeds typical prediction
    gap_measure = max(f_best - med_mu, 0.0)
    # Scale by typical function range for normalization
    func_range = max(med_sigma, 1e-6)
    gap_norm = gap_measure / func_range
    
    # Weight: w_explore increases with gap_norm, capped
    w_explore = 0.3 + 0.4 * min(gap_norm, 1.0)  # range [0.3, 0.7]
    w_exploit = 1.0 - w_explore
    
    # Blended normalized score
    blended = w_exploit * exploit_norm + w_explore * explore_norm
    
    # Combine: use UCB as primary, blend in quantile-normalized score
    # Normalize UCB to [0,1] via rank for consistent scale
    ucb_ranks = np.argsort(np.argsort(ucb_score)).astype(np.float64) / max(n - 1, 1)
    
    # Final score: weighted combination of raw UCB (scaled) and quantile blend
    # Use raw UCB scaled to comparable range, plus quantile blend as refinement
    ucb_scale = np.std(ucb_score) + 1e-9
    ucb_scaled = (ucb_score - np.mean(ucb_score)) / ucb_scale  # ~N(0,1)
    
    # Map blended [0,1] to similar scale
    blended_scaled = (blended - 0.5) * 2.0  # [-1, 1]
    
    # Final: primary UCB + secondary quantile refinement
    score = 0.8 * ucb_scaled + 0.2 * blended_scaled
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > 280:
        return score
    
    return score
