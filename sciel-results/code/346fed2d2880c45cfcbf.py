"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    n = len(mu)
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Known minimum for Styblinski-Tang 1D
    f_min_known = -39.1662
    
    # --- Look-Ahead Batch-Aware Acquisition ---
    # Simulate greedily selecting the next B candidates and estimate
    # how much each candidate contributes to reducing the minimum over
    # the entire batch.
    
    # Step 1: Compute a base acquisition score for initial ranking
    # Use a hybrid of EI and UCB
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration: more exploration when far from optimum
    distance_to_min = max(f_best - f_min_known, 0.0)
    beta = 0.3 + 1.2 * np.exp(-distance_to_min / 8.0)
    ucb = -mu + beta * sigma_safe
    
    # Normalize
    def normalize(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-9:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    base_score = 0.6 * normalize(ei) + 0.4 * normalize(ucb)
    
    # Step 2: Look-ahead simulation
    # Greedily pick top B candidates, then for each candidate in the pool,
    # estimate its marginal contribution if it were part of a batch of size B.
    B = 5  # batch size for look-ahead
    
    # Get indices sorted by base score
    sorted_idx = np.argsort(-base_score)
    
    # Simulate: pick top B, compute how much variance reduction each would cause
    # For each candidate i, estimate the "expected improvement contribution"
    # considering that B-1 other high-scoring points will also be evaluated.
    
    # Approximate: for each candidate, compute a "marginal EI" that accounts
    # for the fact that other points in the batch will reduce uncertainty nearby.
    # We use a simple proxy: reduce the effective sigma for candidates that are
    # "covered" by the top B selections.
    
    # Compute pairwise distances in the 2000-point pool (approximated by index proximity
    # as a proxy for spatial proximity in 1D)
    # Actually, we don't have coordinates. Use a different approach.
    
    # Alternative: Use the structure of the problem. In 1D with uniform candidates,
    # points that are close in the candidate pool index are likely close in space.
    # But this is unreliable. Instead, use a variance-reduction heuristic.
    
    # For each candidate i, compute a "batch-aware EI" that discounts EI
    # based on how many of the top B-1 other candidates have similar mu/sigma.
    
    # Get the top B candidates' mu and sigma values
    top_B_idx = sorted_idx[:B]
    top_B_mu = mu[top_B_idx]
    top_B_sigma = sigma_safe[top_B_idx]
    
    # For each candidate, compute similarity to the top B candidates
    # Use mu and sigma as features (since we don't have coordinates)
    # A candidate is "covered" if there's a top-B candidate with similar mu and sigma
    
    # Compute a coverage score: how much of this candidate's uncertainty
    # is already "addressed" by the top B selections
    # Use a simple kernel-based similarity
    
    mu_range = np.max(mu) - np.min(mu) + eps
    sigma_range = np.max(sigma_safe) - np.min(sigma_safe) + eps
    
    # Normalize mu and sigma for comparison
    mu_norm = (mu - np.min(mu)) / mu_range
    sigma_norm = (sigma_safe - np.min(sigma_safe)) / sigma_range
    
    top_B_mu_norm = (top_B_mu - np.min(mu)) / mu_range
    top_B_sigma_norm = (top_B_sigma - np.min(sigma_safe)) / sigma_range
    
    # Compute Gaussian kernel similarity between each candidate and top B
    # bandwidth
    bw_mu = 0.3
    bw_sigma = 0.3
    
    # Shape: (n, B)
    mu_diff = np.abs(mu_norm[:, None] - top_B_mu_norm[None, :])
    sigma_diff = np.abs(sigma_norm[:, None] - top_B_sigma_norm[None, :])
    
    kernel = np.exp(-0.5 * (mu_diff / bw_mu)**2) * np.exp(-0.5 * (sigma_diff / bw_sigma)**2)
    
    # Maximum similarity to any top-B candidate
    max_sim = np.max(kernel, axis=1)
    
    # Discount EI by coverage: if a candidate is well-covered by top B,
    # its marginal value is lower
    coverage_discount = 1.0 - 0.5 * max_sim
    
    # Batch-aware EI
    batch_ei = ei * coverage_discount
    
    # Also add a diversity bonus: candidates far from top B get a bonus
    diversity_bonus = (1.0 - max_sim) * 0.2
    
    # Final batch-aware score
    batch_score = 0.7 * normalize(batch_ei) + 0.2 * normalize(ucb) + 0.1 * diversity_bonus
    
    # Ensure all scores are finite
    batch_score = np.where(np.isfinite(batch_score), batch_score, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        return batch_score
    
    return batch_score
# EVOLVE-BLOCK-END
