"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Improved Quantile-based exploration strategy:
    1. Compute empirical percentile ranks of sigma.
    2. Identify the top 10% of candidates by uncertainty.
    3. Filter for candidates with predicted mean below the current best (f_best).
    4. If such candidates exist, select the one with the lowest mean (highest -mu score).
       Assign a very high score to these and a very low score to others.
    5. If no such candidates exist, fallback to a UCB-like score to ensure progress.
    """
    n = len(mu)
    
    # Compute percentile ranks of sigma (0 to 1, higher = more uncertain)
    # Using argsort of argsort is O(N log N) and robust
    sigma_ranks = np.argsort(np.argsort(sigma)).astype(np.float64) / (n - 1)
    
    # Define the top 10% threshold (wider than 5% to have more options)
    tau = 0.90
    
    # Identify candidates in the top uncertainty band
    in_top_band = sigma_ranks >= tau
    
    # Identify candidates with promising means (better than current best)
    below_best = mu < f_best
    
    # Valid candidates: High uncertainty AND promising mean
    valid = in_top_band & below_best
    
    scores = np.full(n, -1e9, dtype=np.float64)
    
    if np.any(valid):
        # Among valid candidates, prioritize the lowest mean (best predicted value)
        # We use -mu as the score. Since we already filtered for high uncertainty,
        # this effectively picks the "most optimistic" point in the high-uncertainty region.
        scores[valid] = -mu[valid]
    else:
        # Fallback: No high-uncertainty point has a mean better than f_best.
        # This can happen if the GP is overconfident or we are in a flat region.
        # Use a standard UCB-like score to continue exploration.
        # We use a conservative beta to avoid jumping too far.
        beta = 1.5
        scores = -mu + beta * sigma
        
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
