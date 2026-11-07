"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Batch-aware Look-Ahead acquisition function.
    
    Simulates the next 5 evaluations by greedily selecting top-scoring candidates
    from the pool, updating posterior variance locally to approximate information gain,
    and scoring candidates based on their expected contribution to reducing the minimum value.
    """
    n = len(mu)
    
    # Initialize working copies
    mu_work = mu.copy().astype(np.float64)
    sigma_work = sigma.copy().astype(np.float64)
    
    # Local approximation of variance reduction:
    # When we "observe" a point, the posterior variance decreases.
    # For a Matern-5/2 kernel with length scale 0.2 on [0,1]^1,
    # the correlation between nearby points is high.
    # We approximate the variance update by reducing sigma at nearby points.
    
    # Precompute a distance-based weight matrix approximation.
    # Since we don't have coordinates, we use the index as a proxy for spatial position.
    # This is a heuristic: assume the 2000 candidates are roughly uniformly distributed
    # in the 1D domain, so index order approximates spatial order.
    
    # Scale: 2000 points over [-5, 5], so spacing is 10/2000 = 0.005
    # Length scale is 0.2 on [0,1] which corresponds to 0.2*10 = 2.0 in original units
    # So the kernel length scale in index space is 2.0 / 0.005 = 400 indices
    
    # For Matern-5/2: k(r) = (1 + sqrt(3)*r + 3*r^2/3) * exp(-sqrt(3)*r) where r = |x-x'|/l
    # In index space, r = |i-j| / 400
    
    # We'll use a simpler local variance update:
    # When point k is selected, reduce sigma at point i by a factor based on correlation.
    # The variance of the posterior at i given observation at k is:
    # sigma_i^2_new = sigma_i^2 * (1 - corr(i,k)^2) approximately
    # where corr(i,k) is the kernel correlation.
    
    # For efficiency, we only update a local neighborhood around the selected point.
    # Use a window of +/- 200 indices (half the length scale).
    
    window = 200
    batch_size = 5
    
    # Compute base EI scores for initial selection
    z = (f_best - mu_work) / (sigma_work + 1e-9)
    scores = (f_best - mu_work) * norm.cdf(z) + sigma_work * norm.pdf(z)
    
    # Track which points are "used"
    used = np.zeros(n, dtype=bool)
    
    # Look-ahead simulation
    for step in range(batch_size):
        # Find the best candidate among unused points
        masked_scores = np.where(used, -np.inf, scores)
        best_idx = np.argmax(masked_scores)
        
        # Mark as used
        used[best_idx] = True
        
        # Update variance locally: reduce sigma for nearby points
        # Compute correlation with the selected point
        i_start = max(0, best_idx - window)
        i_end = min(n, best_idx + window + 1)
        
        indices = np.arange(i_start, i_end)
        dists = np.abs(indices - best_idx)
        # Normalize by length scale in index space (400)
        r = dists / 400.0
        # Matern-5/2 correlation
        sqrt3 = np.sqrt(3.0)
        corr = (1.0 + sqrt3 * r + r**2) * np.exp(-sqrt3 * r)
        
        # Update sigma: sigma_new = sigma * sqrt(1 - corr^2)
        # But only for unused points
        mask_unused = ~used[i_start:i_end]
        if np.any(mask_unused):
            sigma_work[i_start:i_end][mask_unused] *= np.sqrt(1.0 - corr[mask_unused]**2 + 1e-12)
        
        # Recompute scores for affected region
        z_local = (f_best - mu_work[i_start:i_end]) / (sigma_work[i_start:i_end] + 1e-9)
        scores[i_start:i_end] = (f_best - mu_work[i_start:i_end]) * norm.cdf(z_local) + sigma_work[i_start:i_end] * norm.pdf(z_local)
        
        # Also update scores for used points to -inf
        scores[i_start:i_end][used[i_start:i_end]] = -np.inf
    
    # Final scores: for each candidate, compute its expected contribution
    # A candidate is valuable if it's likely to improve f_best in the next few steps.
    # We score each candidate by its EI after the look-ahead simulation,
    # but we want to prioritize candidates that are good NOW and also lead to good future selections.
    
    # Simple approach: return the EI scores after the look-ahead has been simulated.
    # But this doesn't differentiate well. Instead, let's compute for each candidate:
    # - Its own EI (how good it is to evaluate now)
    # - A bonus for being in a region where the look-ahead found good points
    
    # Alternative: score = EI + bonus where bonus reflects how much the local variance
    # was reduced (indicating this point would provide information).
    
    # Let's use a different approach: for each candidate, compute the EI assuming
    # it's evaluated first, then simulate 4 more greedy steps, and average the final f_best.
    # This is O(n * batch_size * n) which is too slow for n=2000.
    
    # Compromise: use the EI from the initial computation, but boost candidates
    # that are close to the top-5 selected in the look-ahead simulation.
    
    # Reset to original sigma for final scoring
    sigma_orig = sigma.copy().astype(np.float64)
    z_orig = (f_best - mu) / (sigma_orig + 1e-9)
    ei_orig = (f_best - mu) * norm.cdf(z_orig) + sigma_orig * norm.pdf(z_orig)
    
    # Boost candidates near the selected top-5
    selected_indices = np.where(used)[0]
    boost = np.zeros(n)
    for si in selected_indices:
        i_start = max(0, si - window)
        i_end = min(n, si + window + 1)
        indices = np.arange(i_start, i_end)
        dists = np.abs(indices - si)
        r = dists / 400.0
        sqrt3 = np.sqrt(3.0)
        corr = (1.0 + sqrt3 * r + r**2) * np.exp(-sqrt3 * r)
        # Boost proportional to correlation (how informative this point is for nearby points)
        boost[i_start:i_end] += corr * 0.1
    
    # Final score: EI + small boost for being in informative regions
    final_scores = ei_orig + boost
    
    # Ensure finite values
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
