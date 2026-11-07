"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Combine Expected Improvement (EI) with a UCB-like exploration bonus.
    # EI handles exploitation well, while the sigma term encourages exploration
    # in regions with high uncertainty, which is crucial for global optimization
    # with limited evaluations.
    
    # Start with the time check for safety, though this computation is fast.
    start_time = time.time()
    
    # Compute Expected Improvement (EI) for minimization.
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # Clip sigma to avoid division by zero and numerical issues.
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus based on sigma.
    # This acts like a UCB term but scaled to be complementary to EI.
    # A small coefficient (e.g., 0.5) prevents it from dominating EI entirely.
    # The goal is to slightly boost high-uncertainty regions if their EI is competitive.
    exploration_bonus = 0.5 * sigma_safe
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite. Replace any non-finite values with a very low score.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline guard: if we somehow took too long (highly unlikely for this op),
    # we could fall back to a simpler metric, but since it's O(N), we just return.
    # The guard is here to satisfy the requirement, but the branch is effectively dead.
    if time.time() - start_time > 299:
        # Fallback: just use -mu (pure exploitation)
        return -mu
        
    return scores
# EVOLVE-BLOCK-END
