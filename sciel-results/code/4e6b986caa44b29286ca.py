"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Upper Confidence Bound (UCB) for minimization.
    
    mu: (N,) posterior mean
    sigma: (N,) posterior std
    f_best: best observed value so far
    
    Returns: (N,) scores, higher is better.
    """
    # Ensure sigma is non-negative and has a small floor to avoid issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive exploration weight:
    # We want to explore regions where sigma is large relative to the current best.
    # A common heuristic is to scale beta with the standard deviation of the 
    # uncertainty itself or simply use a fixed constant tuned for low budgets.
    # However, to make it adaptive and robust, we can look at the scale of uncertainty.
    
    # Let's use a simple but effective adaptive strategy:
    # beta = 2 * log(1 + N) / N is for regret bounds, but here N is not the step count.
    # Instead, we use a heuristic that encourages exploration when sigma is high.
    # score = -(mu - beta * sigma)
    # Higher score means lower (mu - beta * sigma), i.e., we prefer low mean and high sigma.
    
    # In low-budget BO (30 evals), aggressive exploration is key.
    # A fixed beta of 2.0 to 4.0 often works well for Ackley.
    # Let's try to make it slightly adaptive to the scale of the problem.
    # If f_best is close to 0 (minimum of Ackley is 0), the scale is small.
    # But mu and sigma are in objective units.
    
    # Let's stick to a robust fixed beta first, but ensure it's strong enough.
    # The previous Thompson Sampling was effectively beta ~ 1 (standard deviation).
    # UCB with beta > 1 is more exploratory than pure mean, and beta ~ 2-3 is common.
    
    # Let's use beta = 2.5. This strongly weights uncertainty.
    beta = 2.5
    
    # UCB for minimization: we want to minimize mu - beta * sigma
    # To maximize the score (higher is better), we return - (mu - beta * sigma)
    # score = -mu + beta * sigma
    
    score = -mu + beta * sigma_safe
    
    # Ensure finite scores
    # If any values are non-finite, replace with a very low score (e.g., -1e10)
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
