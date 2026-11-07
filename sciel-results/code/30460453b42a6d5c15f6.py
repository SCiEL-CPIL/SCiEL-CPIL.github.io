"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid UCB with Entropy-based exploration.
    
    Args:
        mu: (N,) array of GP means.
        sigma: (N,) array of GP standard deviations.
        f_best: float, the best observed value so far.
        
    Returns:
        scores: (N,) array of acquisition scores (higher is better).
    """
    start_time = time.time()
    deadline = 290  # Safety margin from 300s limit
    
    n = len(mu)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Exploitation Term: Standard Expected Improvement (EI)
    # EI(x) = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # This drives the search towards points likely to improve upon f_best.
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Exploration Term: Entropy-based Information Gain
    # The entropy of a Gaussian N(mu, sigma^2) is 0.5 * log(2 * pi * e * sigma^2).
    # Maximizing entropy encourages visiting regions with high uncertainty.
    # We use the log-entropy to penalize regions with very low sigma (exploited)
    # and reward regions with high sigma (unexploited).
    # log(entropy) ~ log(sigma) + constant.
    # To make it scale-invariant and comparable to EI, we can just use log(sigma).
    # However, EI is in objective units. Log(sigma) is in log-objective units.
    # A common heuristic is to add a term proportional to log(sigma) or sigma itself.
    # Let's use a weighted sum. The weight 'beta' controls exploration.
    # A dynamic beta could be used, but a fixed one is safer for stability.
    # Let's try a term that grows with sigma but is dampened to prevent pure random search.
    
    # Alternative Exploration: Kriging Believer / UCB
    # UCB = mu + kappa * sigma
    # But for minimization, we want low mu and high sigma.
    # Score = -mu + kappa * sigma
    # This is equivalent to EI with a specific limit.
    
    # Let's combine EI (which is robust) with a Log-Entropy bonus.
    # The entropy bonus helps escape local minima by explicitly rewarding uncertainty.
    # We normalize the entropy term to have similar magnitude to EI on average.
    # Standard deviation of mu and sigma across candidates can help, but 
    # global normalization is risky if the GP is bad.
    
    # Simple robust combination:
    # Score = EI + beta * log(1 + sigma)
    # Using log(1+sigma) prevents -inf for sigma=0 and dampens very large sigma.
    
    beta = 1.0  # Exploration weight
    
    # Check time
    if time.time() > deadline - 5:
        # Fallback: Pure EI if we are close to deadline (calculation is fast anyway)
        return ei
        
    # Calculate exploration bonus
    # We want to encourage high sigma. 
    # log(1 + sigma) is a good smooth proxy for entropy.
    exploration_bonus = beta * np.log1p(sigma_safe)
    
    scores = ei + exploration_bonus
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
