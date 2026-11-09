"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Batch-aware, diversity-penalized acquisition function.
    
    Uses Expected Improvement as the base score.
    Computes a diversity penalty based on the maximum Matern-5/2 kernel
    correlation with a set of top-K candidates in the (mu, sigma) feature space.
    This penalizes candidates that are "redundant" with other high-potential
    candidates, encouraging exploration of distinct regions.
    """
    n = len(mu)
    
    # 1. Base Acquisition: Expected Improvement
    # Handle sigma=0 cases safely
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (standard EI is always >= 0)
    ei = np.maximum(ei, 0)
    
    # 2. Select Top-K candidates for diversity calculation
    K = 50
    # If n is smaller than K, use all
    K_eff = min(K, n)
    top_k_indices = np.argsort(ei)[-K_eff:]
    
    # 3. Prepare features for distance calculation
    # We use mu and sigma as proxies for location in the latent space.
    # Normalize them to have zero mean and unit variance to make distance scale-invariant.
    mu_mean = np.mean(mu)
    mu_std = np.std(mu) + 1e-9
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + 1e-9
    
    mu_norm = (mu - mu_mean) / mu_std
    sigma_norm = (sigma - sigma_mean) / sigma_std
    
    # Features shape: (n, 2)
    features = np.column_stack((mu_norm, sigma_norm))
    
    # Extract features for top-K candidates
    top_k_features = features[top_k_indices]
    
    # 4. Compute distances from all candidates to Top-K candidates
    # dists[i, j] = distance between candidate i and top-k candidate j
    # Using broadcasting: (n, 1, 2) - (1, K, 2) -> (n, K, 2)
    diff = features[:, np.newaxis, :] - top_k_features[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)  # Shape (n, K)
    
    # 5. Convert distances to Matern-5/2 correlations
    # Kernel: K(r) = (1 + sqrt(3)*r + 3/5*r^2) * exp(-sqrt(3)*r)
    # We need a length scale for the normalized feature space.
    # A reasonable length scale for standardized features is around 0.5 to 1.0.
    # Let's estimate a length scale from the Top-K distances to be adaptive.
    # Median distance between top-K points might be a good indicator of "crowding" scale.
    if K_eff > 1:
        # Compute pairwise distances among top-K points
        top_k_diff = top_k_features[:, np.newaxis, :] - top_k_features[np.newaxis, :, :]
        top_k_dists = np.linalg.norm(top_k_diff, axis=2)
        # Get upper triangle to avoid zeros
        mask = np.triu(np.ones((K_eff, K_eff), dtype=bool), k=1)
        non_zero_dists = top_k_dists[mask]
        if len(non_zero_dists) > 0:
            length_scale = np.median(non_zero_dists)
        else:
            length_scale = 0.5
    else:
        length_scale = 0.5
        
    # Ensure length_scale is not too small
    length_scale = max(length_scale, 0.1)
    
    r = dists / length_scale
    sqrt3 = np.sqrt(3)
    # Matern-5/2 kernel value
    kern = (1 + sqrt3 * r + 0.6 * r**2) * np.exp(-sqrt3 * r)
    
    # 6. Compute penalty: Maximum correlation with any top-K candidate
    # We want to penalize if a candidate is very similar to ANY of the top-K candidates.
    # Note: If the candidate itself is in the top-K, its distance to itself is 0, so kern=1.
    # We should exclude self-correlation if we want to penalize redundancy with *others*.
    # However, the directive says "maximum pairwise correlation among the top-K candidates".
    # This usually implies we look at the set of top-K, and for any point, we see how close it is to that set.
    # If a point is in the top-K, it is "redundant" with itself? No, we want to select distinct points.
    # So, for a candidate i, we calculate max correlation with {j in Top-K | j != i}.
    
    # Create a mask to exclude self if the candidate is in top_k_indices
    # top_k_indices is a set of indices.
    # For each candidate i, if i is in top_k_indices, we need to mask out the column corresponding to i.
    
    # Efficient way:
    # Initialize max_corr to 0
    max_corr = np.zeros(n)
    
    # We can vectorize the "exclude self" part by setting self-correlation to 0 before taking max.
    # But we don't have a direct (n, n) matrix, we have (n, K).
    # Let's create a mask matrix (n, K) where mask[i, j] is True if candidate i is the same as top_k candidate j.
    
    # top_k_indices is an array of length K_eff.
    # We need to check if features[i] is the same as top_k_features[j].
    # Since indices are unique, we can check if i is in top_k_indices.
    
    # Let's just compute max_corr normally, then for the top-K candidates, recompute excluding self.
    # Actually, simpler:
    # For all i, max_corr[i] = max(kern[i, :])
    # For i in top_k_indices, we need max(kern[i, :]) excluding the column where top_k_indices[j] == i.
    
    max_corr = np.max(kern, axis=1)
    
    # Correct for self-correlation for the top-K candidates
    for i in top_k_indices:
        # Find the column index j in top_k_indices where top_k_indices[j] == i
        # Since top_k_indices are unique, we can use np.where
        j_idx = np.where(top_k_indices == i)[0][0]
        # Create a copy of kern[i] with j_idx set to 0 (or -inf)
        kern_row = kern[i].copy()
        kern_row[j_idx] = 0.0
        max_corr[i] = np.max(kern_row)
        
    # 7. Final Score
    # Penalty is proportional to max_corr.
    # We want to subtract this.
    # Scale the penalty to be comparable to EI.
    # EI is in objective units. max_corr is in [0, 1].
    # We need a scaling factor. 
    # A common choice is to scale by the range of EI or a fraction of f_best.
    # Let's use a fraction of the max EI or a fixed value relative to f_best.
    # If f_best is 0 (minimum), EI can be large.
    # Let's use a dynamic scale: penalty_scale = 0.5 * np.max(ei)
    # Or simply a fixed coefficient if EI is well-behaved.
    # Given the "standardized targets" mention in GP, but mu/sigma are in objective units.
    # Let's use a coefficient of 0.5 for the correlation penalty.
    
    penalty_scale = 0.5
    scores = ei - penalty_scale * max_corr
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
