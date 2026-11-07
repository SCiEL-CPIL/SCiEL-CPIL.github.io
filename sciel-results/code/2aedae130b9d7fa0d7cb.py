"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are close to the 330s limit, return a fast, safe fallback
    if time.time() - _start_time > 325:
        safe_cost = np.maximum(cost, 1e-6)
        return -mu / safe_cost

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    safe_cost = np.maximum(cost, 1e-6)
    
    # Compute z-score for Expected Improvement
    # For minimization, improvement is f_best - y
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical issues in norm.cdf and norm.pdf
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Clip EI to be non-negative (mathematically it should be >= 0, but numerical errors can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Thinning-based EI: approximate Knowledge Gradient by subtracting
    # the expected improvement of the top-k candidates (k=5) from each candidate's EI.
    # This models the probability that the current point will remain the best after a batch.
    # We approximate this by sorting candidates by EI and subtracting a fraction of the top-k EI values.
    # Specifically, for each candidate i, we subtract the average EI of the top-5 candidates
    # (excluding i if i is in the top 5) scaled by a factor.
    
    k = 5
    # Get indices of top-k EI values
    top_k_indices = np.argpartition(ei, -k)[-k:] if ei.size >= k else np.arange(ei.size)
    
    # Compute the mean EI of the top-k candidates
    top_k_ei_mean = np.mean(ei[top_k_indices])
    
    # For each candidate, if it is in the top-k, we don't subtract its own EI
    # but we subtract the mean of the other top-k-1 candidates.
    # Otherwise, we subtract the full top-k mean.
    # This is a simplified approximation of the KG thinning effect.
    
    # Create a mask for candidates in the top-k
    in_top_k = np.zeros_like(ei, dtype=bool)
    in_top_k[top_k_indices] = True
    
    # For candidates in the top-k, subtract the mean of the other top-k-1 EIs
    # For candidates not in the top-k, subtract the full top-k mean
    # This encourages exploration of points that are not already among the best
    # but still have significant EI, balancing exploitation and exploration.
    
    # Simplified: subtract top_k_ei_mean from all, but add back a fraction for top-k candidates
    # to avoid zeroing them out completely.
    # Actually, a better approximation: the KG value is EI_i - E[max(0, EI_j for j != i)]
    # We approximate E[max(0, EI_j)] by the mean of the top-k EIs (excluding i if possible).
    
    # For simplicity and speed, we use a soft penalty:
    # score_i = EI_i - alpha * top_k_ei_mean, where alpha is a small constant.
    # But to be more principled, let's do:
    # If i is in top-k, penalty = mean(ei[top_k] excluding i)
    # Else, penalty = mean(ei[top_k])
    
    if len(top_k_indices) > 1:
        penalty = np.full_like(ei, top_k_ei_mean)
        # For each top-k candidate, compute the mean of the other top-k-1
        for idx in top_k_indices:
            others = [j for j in top_k_indices if j != idx]
            if others:
                penalty[idx] = np.mean(ei[others])
    else:
        penalty = np.zeros_like(ei)
    
    # Apply the penalty
    ei_thinned = ei - penalty
    
    # Ensure non-negative
    ei_thinned = np.maximum(ei_thinned, 0.0)
    
    # Cost-aware score: EI per unit cost
    # Use cost ** 0.8 as a softened penalty (from inspiration 1)
    cost_penalty = safe_cost ** 0.8
    score = ei_thinned / cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
