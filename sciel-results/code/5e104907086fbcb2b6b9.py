"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_TIMEOUT = 290  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function using Monte Carlo approximation.
    
    KG estimates the expected improvement in the global minimum by sampling
    from the posterior predictive at top-k candidates and computing how often
    each candidate would be the new best. This directly optimizes for regret
    reduction rather than pointwise improvement.
    
    Parameters:
    -----------
    mu : ndarray of shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray of shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value so far.
    
    Returns:
    --------
    scores : ndarray of shape (2000,)
        Acquisition scores; higher values indicate more desirable points.
    """
    # Deadline guard: if we are close to the timeout, return a fast fallback
    if time.time() - _START_TIME > _TIMEOUT:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei

    sigma_safe = np.maximum(sigma, 1e-9)
    n = len(mu)
    
    # Select top-k candidates based on a quick EI score
    # This reduces the MC computation to a manageable subset
    z = (f_best - mu) / sigma_safe
    ei_quick = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Top-k for KG computation: use 200 candidates
    k = min(200, n)
    top_k_idx = np.argsort(ei_quick)[-k:]
    
    # Initialize KG scores to zero
    kg_scores = np.zeros(n)
    
    # Monte Carlo samples for KG
    n_samples = 50
    rng = np.random.default_rng(seed=42)
    
    # For each of the top-k candidates, estimate its KG value
    # KG(i) = E[max(f_best - min(f_i, f_other), 0)] where f_i ~ N(mu_i, sigma_i^2)
    # and f_other ~ N(mu_other, sigma_other^2) for other candidates
    
    # To make this tractable, we approximate:
    # For candidate i, sample f_i from N(mu_i, sigma_i^2)
    # For each sample, compute the probability that f_i is better than all others
    # and the expected improvement
    
    # More efficient: use the fact that for two independent normals,
    # P(f_i < f_j) = Phi((mu_j - mu_i) / sqrt(sigma_i^2 + sigma_j^2))
    
    # For top-k, compute pairwise probabilities
    mu_k = mu[top_k_idx]
    sigma_k = sigma_safe[top_k_idx]
    
    # KG for each top-k candidate:
    # KG(i) = E[ max(f_best - min(f_i, min_{j!=i} f_j), 0) ]
    #        = E[ max(f_best - f_i, 0) * P(f_i <= min_{j!=i} f_j) ]
    #          + E[ max(f_best - min_{j!=i} f_j, 0) * P(f_i > min_{j!=i} f_j) ]
    
    # Simplified: KG(i) ≈ E[ max(f_best - f_i, 0) ] * P(f_i is minimum among top-k)
    #              + (1 - P(f_i is minimum)) * (f_best - mu_min_others) * P(f_best > mu_min_others)
    
    # Even simpler approximation using Monte Carlo:
    # Sample f_i for each top-k candidate, find the minimum, compute improvement
    
    # Vectorized MC approach:
    # For each sample, draw f_i ~ N(mu_i, sigma_i^2) for all k candidates
    # The minimum of these is the best found
    # KG(i) contribution: how often is candidate i the minimum, and what's the improvement
    
    # Let's do a proper MC:
    # For s in 1..n_samples:
    #   Draw f_s ~ N(mu_k, sigma_k^2) for all k candidates
    #   min_val_s = min(f_s)
    #   improvement_s = max(f_best - min_val_s, 0)
    #   For each i, weight_s[i] = 1 if i is argmin, else 0
    #   KG[i] += weight_s[i] * improvement_s
    # KG[i] /= n_samples
    
    # But this gives KG only for top-k. For others, use EI as fallback.
    
    # Actually, a better MC approach:
    # KG(i) = E[ max(f_best - min(f_i, f_other), 0) ]
    # where f_other is the minimum of all other candidates
    # 
    # Approximate f_other by the minimum of the top-k\{i} candidates
    
    # Let's use a different approach: compute for each top-k candidate i:
    # Sample f_i ~ N(mu_i, sigma_i^2)
    # Sample f_j ~ N(mu_j, sigma_j^2) for j in top-k\{i}
    # min_others = min(f_j for j != i)
    # improvement = max(f_best - min(f_i, min_others), 0)
    # KG[i] = average over samples
    
    # This is O(k * n_samples * k) which is fine for k=200, n_samples=50
    
    kg_top_k = np.zeros(k)
    
    # Pre-compute: for each i, the indices of other candidates
    # Use vectorized sampling
    
    # Sample all k candidates n_samples times: shape (n_samples, k)
    samples = rng.normal(loc=mu_k[None, :], scale=sigma_k[None, :], size=(n_samples, k))
    
    # For each sample s, for each candidate i:
    # min_others_s_i = min(samples[s, j] for j != i)
    # This is expensive to do naively. Use a trick:
    # min_others_s_i = min(samples[s]) if samples[s, i] != min(samples[s]), else second_min(samples[s])
    
    # Compute min and second min for each sample
    # Sort each sample row
    sorted_samples = np.sort(samples, axis=1)  # shape (n_samples, k)
    min_val = sorted_samples[:, 0]  # shape (n_samples,)
    second_min = sorted_samples[:, 1] if k > 1 else np.full(n_samples, np.inf)
    
    # For each sample s and candidate i:
    # If samples[s, i] == min_val[s], then min_others = second_min[s]
    # Else min_others = min_val[s]
    
    # Create a mask: is candidate i the minimum in sample s?
    # min_mask[s, i] = (samples[s, i] == min_val[s])
    # But floating point comparison is tricky. Use argmin instead.
    
    argmin_idx = np.argmin(samples, axis=1)  # shape (n_samples,)
    
    # For each sample s, the minimum is at argmin_idx[s]
    # For candidate i, min_others[s, i] = 
    #   second_min[s] if i == argmin_idx[s], else min_val[s]
    
    # Build min_others matrix: shape (n_samples, k)
    min_others = np.where(
        np.arange(k)[None, :] == argmin_idx[:, None],
        second_min[:, None],
        min_val[:, None]
    )
    
    # Now for each sample s and candidate i:
    # min_combined = min(samples[s, i], min_others[s, i])
    # improvement = max(f_best - min_combined, 0)
    # KG[i] += improvement / n_samples
    
    min_combined = np.minimum(samples, min_others)  # shape (n_samples, k)
    improvements = np.maximum(f_best - min_combined, 0)  # shape (n_samples, k)
    
    kg_top_k = np.mean(improvements, axis=0)  # shape (k,)
    
    # Assign KG scores to top-k candidates
    kg_scores[top_k_idx] = kg_top_k
    
    # For non-top-k candidates, use EI as a proxy
    z_all = (f_best - mu) / sigma_safe
    ei_all = (f_best - mu) * norm.cdf(z_all) + sigma_all_safe * norm.pdf(z_all) if False else (f_best - mu) * norm.cdf(z_all) + sigma_safe * norm.pdf(z_all)
    
    # Scale EI to match the scale of KG scores
    # KG scores are in objective units (improvement), EI is also in objective units
    # But KG tends to be larger because it considers the minimum over many candidates
    # Scale EI by a factor to make them comparable
    if np.max(kg_top_k) > 1e-9:
        ei_scaled = ei_all * (np.max(kg_top_k) / (np.max(ei_all) + 1e-9))
    else:
        ei_scaled = ei_all
    
    # For non-top-k, use scaled EI
    non_top_mask = np.ones(n, dtype=bool)
    non_top_mask[top_k_idx] = False
    kg_scores[non_top_mask] = ei_scaled[non_top_mask]
    
    # Ensure finite values
    scores = np.where(np.isfinite(kg_scores), kg_scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
