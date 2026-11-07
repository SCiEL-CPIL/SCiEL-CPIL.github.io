"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Batch-aware Look-Ahead acquisition function.
    
    For the top M candidates, simulates the next K-1 evaluations by greedily
    selecting the highest EI point from the remaining pool, updating the
    posterior variance locally (assuming observed value equals the mean),
    and summing the expected improvements of the simulated sequence.
    This rewards candidates that not only have high immediate EI but also
    lead to a sequence of high-EI future evaluations.
    """
    n = len(mu)
    
    # Compute base EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Identify top M candidates to evaluate with look-ahead
    M = 50
    K = 5  # Total batch size to simulate
    
    # Get indices of top M candidates
    top_m_indices = np.argsort(ei)[-M:]
    
    # Precompute coordinates in [0, 1] for kernel calculation
    # x ranges from -5 to 5, n points
    x_scaled = np.arange(n) / (n - 1)
    
    # Function to compute Matern 5/2 correlation between two indices
    def matern_corr(i, j):
        # Vectorized for i fixed, j array
        d = np.abs(x_scaled[i] - x_scaled[j])
        sqrt3 = np.sqrt(3.0)
        return (1.0 + sqrt3 * d + 1.5 * d**2) * np.exp(-sqrt3 * d)
    
    # Initialize scores with base EI
    scores = ei.copy()
    
    # For each top candidate, simulate the rest of the batch
    for i in top_m_indices:
        # Copy current state
        mu_sim = mu.copy()
        sigma_sim = sigma.copy()
        
        # Mark i as "observed" (conceptually)
        # Update variance for all j != i
        # corr[i] = 1, so sigma[i] becomes 0.
        # For j != i: sigma[j] *= sqrt(1 - corr[i,j]^2)
        
        corr_i = matern_corr(i, np.arange(n))
        # Ensure numerical stability
        corr_sq = np.clip(corr_i**2, 0, 1)
        
        # Update sigma for all points except i
        # We don't update mu because we assume observed value == mean
        sigma_sim *= np.sqrt(1.0 - corr_sq + 1e-12)
        sigma_sim[i] = 0.0
        
        # Simulate K-1 more steps
        future_ei_sum = 0.0
        used = np.zeros(n, dtype=bool)
        used[i] = True
        
        for step in range(K - 1):
            # Compute EI for remaining points
            z_sim = (f_best - mu_sim) / (sigma_sim + 1e-9)
            ei_sim = (f_best - mu_sim) * norm.cdf(z_sim) + sigma_sim * norm.pdf(z_sim)
            
            # Mask used points
            ei_sim[used] = -np.inf
            
            # Find best
            j_best = np.argmax(ei_sim)
            
            if ei_sim[j_best] == -np.inf:
                break
                
            future_ei_sum += ei_sim[j_best]
            used[j_best] = True
            
            # Update variance for next step
            # Only update points not yet used
            corr_j = matern_corr(j_best, np.arange(n))
            corr_sq_j = np.clip(corr_j**2, 0, 1)
            
            # Apply update only to unused points? 
            # Technically, the variance update applies to all, but used points don't matter.
            # For efficiency, we can update all, but it's O(N).
            sigma_sim *= np.sqrt(1.0 - corr_sq_j + 1e-12)
            sigma_sim[j_best] = 0.0
            
        # Total score for candidate i is immediate EI + future EI
        scores[i] = ei[i] + future_ei_sum
        
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
