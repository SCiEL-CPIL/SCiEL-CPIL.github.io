"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Batch-aware, diversity-penalized acquisition function.
    
    Computes Expected Improvement (EI) for all candidates.
    Then, identifies the top-K candidates by EI.
    For each candidate, computes the maximum similarity (using a Gaussian kernel 
    on normalized (mu, sigma) features) to any candidate in the top-K set.
    Subtracts a penalty proportional to this max similarity from the EI score.
    This discourages selecting candidates that are predictively redundant with 
    the current best candidates, encouraging exploration of distinct regions.
    """
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # 1. Compute Expected Improvement
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle edge cases where EI might be negative or zero
    ei = np.maximum(ei, 0.0)
    
    # 2. Select top-K candidates by EI
    # K is chosen to be small enough to represent the "current focus" 
    # but large enough to enforce diversity. 
    # For 2000 candidates, K=10-20 is reasonable.
    K = 15
    if n < K:
        K = n
        
    top_k_indices = np.argsort(ei)[-K:]
    top_k_mu = mu[top_k_indices]
    top_k_sigma = sigma[top_k_indices]
    
    # 3. Prepare features for similarity computation
    # We use (mu, sigma) as a proxy for location in the latent space.
    # Normalization is crucial.
    
    # Normalize mu and sigma to [0, 1] range or standardize?
    # Standardization (z-score) is better if the scale of mu and sigma differs.
    # However, mu and sigma are in objective units.
    # Let's standardize them independently.
    
    mu_mean = np.mean(mu)
    mu_std = np.std(mu) + 1e-9
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + 1e-9
    
    # Feature vector for all candidates: [mu_z, sigma_z]
    features_all = np.column_stack((
        (mu - mu_mean) / mu_std,
        (sigma - sigma_mean) / sigma_std
    ))
    
    features_top_k = features_all[top_k_indices]
    
    # 4. Estimate length scale for the Gaussian kernel on these features
    # We want the kernel to be sensitive to local structure.
    # Use the median pairwise distance among the top-K features as a heuristic.
    if K > 1:
        # Compute pairwise distances in the 2D feature space
        # dists[i, j] = ||f_i - f_j||
        # We can do this efficiently for small K
        diff = features_top_k[:, np.newaxis, :] - features_top_k[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Get upper triangle to avoid self-distances (0)
        iu = np.triu_indices(K, k=1)
        nonzero_dists = dists[iu]
        if len(nonzero_dists) > 0:
            length_scale = np.median(nonzero_dists)
        else:
            length_scale = 1.0
    else:
        length_scale = 1.0
    
    # Avoid division by zero if all top-K are identical
    length_scale = max(length_scale, 1e-6)
    
    # 5. Compute similarity (Gaussian kernel) between all candidates and top-K candidates
    # Similarity S(i, j) = exp(-||f_i - f_j||^2 / (2 * length_scale^2))
    
    # To be efficient, we compute distances from all N points to the K top points.
    # N=2000, K=15. This is 2000*15 = 30000 operations. Very fast.
    
    # diff_all: (N, K, 2)
    diff_all = features_all[:, np.newaxis, :] - features_top_k[np.newaxis, :, :]
    dists_all = np.sqrt(np.sum(diff_all**2, axis=2))
    
    # Compute kernel values
    # Note: The standard Gaussian kernel is exp(-d^2 / (2*l^2))
    kernel_values = np.exp(-dists_all**2 / (2 * length_scale**2))
    
    # 6. Determine penalty for each candidate
    # Penalty is the maximum similarity to any of the top-K candidates.
    # However, for candidates that ARE in the top-K, the max similarity will be 1.0 (to themselves).
    # We want to penalize redundancy. 
    # If a candidate is in the top-K, it is already "selected" in our mental model of the batch.
    # The penalty should discourage *other* candidates from being too close to these.
    # But since we are returning a score for *all* candidates to pick the *next* single one,
    # the top-K are just a reference set.
    # If we pick the best EI candidate, it will have a penalty of 1.0 (if it's in top-K, which it is).
    # This would heavily penalize the best EI candidate, which might be bad.
    
    # Refinement: The penalty should be based on similarity to *other* top-K candidates.
    # For a candidate $i$, if $i \in \text{top-K}$, we look at max similarity to $\text{top-K} \setminus \{i\}$.
    # If $i \notin \text{top-K}$, we look at max similarity to all $\text{top-K}$.
    
    # Let's create a mask or adjust the kernel values.
    # For each candidate i (row in kernel_values), we want the max over j (columns, top-K indices).
    # If candidate i corresponds to top_k_indices[j], we should exclude column j.
    
    # Map: candidate index -> position in top_k_indices, or -1 if not in top-K
    # We can create an array of penalties.
    
    penalties = np.zeros(n)
    
    # For efficiency, we can compute max similarity for all, then fix the top-K ones.
    max_sim_all = np.max(kernel_values, axis=1)
    
    # For candidates in top-K, we need to exclude the self-similarity (which is 1.0).
    # Let's identify which rows in kernel_values correspond to top-K candidates.
    # The rows are indexed by the original candidate index.
    # top_k_indices contains the original indices.
    
    # We can simply iterate over the top-K candidates to fix their penalties.
    # K is small (15), so this is fast.
    
    for k_idx, orig_idx in enumerate(top_k_indices):
        # Get the row for this candidate
        row = kernel_values[orig_idx]
        # Set the self-similarity (column k_idx) to 0
        row_copy = row.copy()
        row_copy[k_idx] = 0.0
        # The penalty is the max similarity to the *other* top-K candidates
        penalties[orig_idx] = np.max(row_copy)
        
    # For candidates NOT in top-K, the penalty is the max similarity to *all* top-K candidates.
    # We already computed max_sim_all. 
    # We need to set penalties for non-top-K candidates to max_sim_all.
    
    # Create a mask for non-top-K candidates
    is_top_k = np.zeros(n, dtype=bool)
    is_top_k[top_k_indices] = True
    non_top_k_mask = ~is_top_k
    
    penalties[non_top_k_mask] = max_sim_all[non_top_k_mask]
    
    # 7. Apply penalty to EI
    # Tune the penalty strength.
    # If lambda is too high, we ignore EI entirely.
    # If lambda is too low, we don't enforce diversity.
    # A value around 0.5 - 1.0 of the scale of EI might be appropriate.
    # Since EI can be large, and penalty is in [0, 1], we might need to scale EI or penalty.
    # Let's normalize EI to have a similar magnitude to the penalty, or just use a fixed lambda.
    # EI is in objective units. The penalty is a dimensionless correlation.
    # It is safer to scale EI by its own standard deviation or max value to make the penalty comparable.
    
    # Let's normalize EI to [0, 1] range for the subtraction, then scale back?
    # No, the output score just needs to be relative.
    # Let's use: Score = EI - lambda * penalty * EI_scale
    # Where EI_scale is a characteristic magnitude of EI, e.g., np.max(ei) or np.std(ei).
    
    ei_scale = np.max(ei)
    if ei_scale < 1e-9:
        ei_scale = 1.0
        
    lambda_penalty = 0.5  # Tuning parameter
    
    scores = ei - lambda_penalty * penalties * ei_scale
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
