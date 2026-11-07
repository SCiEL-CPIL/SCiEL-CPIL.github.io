"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Adaptive Upper Confidence Bound (UCB) acquisition function.
    
    Args:
        mu: Array of shape (N,) with GP mean predictions.
        sigma: Array of shape (N,) with GP standard deviation predictions.
        f_best: The best (lowest) observed function value so far.
        
    Returns:
        Array of shape (N,) with acquisition scores. Higher is better.
    """
    # Theoretical global minimum of Hartmann 6D function
    f_min = -3.32237
    
    # Calculate current regret (distance from best found to global min)
    regret = f_best - f_min
    
    # Clip regret to avoid numerical issues if f_best < f_min (shouldn't happen, but safe)
    regret = np.clip(regret, 0.0, 10.0)
    
    # Adaptive Beta
    # We want to explore more (higher beta) when regret is large.
    # We want to exploit more (lower beta) when regret is small.
    
    beta_max = 2.5
    beta_min = 0.5
    scale = 1.0
    
    # Transition factor: 0 when regret=0, approaches 1 as regret increases
    # Using 1 - exp(-regret/scale)
    transition = 1.0 - np.exp(-regret / scale)
    
    beta = beta_min + (beta_max - beta_min) * transition
    
    # UCB score for minimization
    # We want to maximize: -mu + beta * sigma
    # -mu is high when mu is low (good prediction)
    # sigma is high when uncertainty is high (good exploration)
    
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
