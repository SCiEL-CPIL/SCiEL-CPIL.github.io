"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Adaptive UCB acquisition function for minimization.
    
    Args:
        mu: Array of posterior means.
        sigma: Array of posterior standard deviations.
        f_best: The best (lowest) function value observed so far.
        
    Returns:
        Array of scores, higher is better.
    """
    # The domain is [-5, 5], so max f(x) is 25.
    # f_best is in [0, 25].
    
    # Normalize f_best to [0, 1] to adjust exploration vs exploitation.
    # High f_best (bad) -> High exploration (high beta).
    # Low f_best (good) -> Low exploration (low beta).
    f_norm = np.clip(f_best / 25.0, 0.0, 1.0)
    
    # Beta ranges from 0.1 (exploit) to 3.0 (explore).
    beta_min = 0.1
    beta_max = 3.0
    beta = beta_min + (beta_max - beta_min) * f_norm
    
    # UCB score for minimization: -mu + beta * sigma
    # We maximize this score.
    scores = -mu + beta * sigma
    
    # Handle numerical stability
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
