"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level RNG for Thompson Sampling to avoid re-initialization overhead
_rng = np.random.default_rng()

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Args:
        mu: numpy array of shape (2000,) posterior means.
        sigma: numpy array of shape (2000,) posterior standard deviations.
        f_best: float, the best observed value so far (not directly used in TS, 
                but kept for interface compatibility).
                
    Returns:
        numpy array of shape (2000,) scores. Higher score means higher priority.
    """
    # Deadline guard: although this operation is very fast, we adhere to the directive.
    # The evaluator kills after 330s. We assume a generous per-call budget.
    # We don't actually need a complex loop here, just a check if we were doing heavy search.
    # For TS, it's O(N).
    
    # Generate standard normal samples
    # Using a module-level RNG ensures we don't pay the cost of creating a new Generator object every call
    z = _rng.standard_normal(size=mu.shape)
    
    # Sample from the posterior: y ~ N(mu, sigma^2)
    # y = mu + sigma * z
    samples = mu + sigma * z
    
    # We are minimizing the objective function.
    # We want the candidate with the LOWEST sample value to have the HIGHEST score.
    # Therefore, we negate the samples.
    scores = -samples
    
    # Ensure finiteness (sigma and mu should be finite, but just in case of numerical weirdness)
    # If any score is NaN or Inf, replace it with a very low value so it's never picked.
    # However, the problem states "Return 2000 finite scores".
    # If mu or sigma has NaN/Inf, the result will too. We should sanitize.
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
# EVOLVE-BLOCK-END
