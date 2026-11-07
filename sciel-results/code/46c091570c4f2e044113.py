"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Draws a single sample from the posterior at each candidate point:
    f_sample = mu + sigma * z, where z ~ N(0, 1).
    Returns -f_sample as the score, so that the candidate with the 
    lowest sampled function value (best potential minimum) gets the 
    highest score.
    """
    # Generate standard normal noise for each candidate point
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sample from the posterior
    # mu and sigma are expected to be in objective units (minimization)
    f_sample = mu + sigma * z
    
    # We want to minimize the function, so lower f_sample is better.
    # The acquisition score should be higher for better candidates.
    # Therefore, we return the negative of the sampled value.
    scores = -f_sample
    
    # Ensure finite scores (though they should be if mu/sigma are)
    # If any are not finite, replace with a very large negative number (bad score)
    # or handle appropriately. The prompt says "Return 2000 finite scores".
    # If sigma is 0, z*sigma is 0. If mu is inf, that's a problem in GP.
    # Assuming valid GP output, scores are finite.
    # Just in case, we can clip or replace non-finite values.
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        # If we have -inf, that's technically not finite. 
        # Let's replace non-finite with a very low value (bad for maximization)
        # or high value? 
        # "higher meaning evaluate next". 
        # If a point is invalid, we don't want to pick it.
        # So we want its score to be low.
        # -inf is low. But the requirement says "finite".
        # Let's use a very large negative number.
        min_val = np.min(scores[np.isfinite(scores)]) if np.any(np.isfinite(scores)) else -1e10
        scores = np.where(np.isfinite(scores), scores, min_val - 1e10)

    return scores
# EVOLVE-BLOCK-END
