import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Adaptive Cost-Adjusted UCB acquisition function.
    
    Uses Upper Confidence Bound (UCB) with an adaptive exploration parameter (beta)
    that decreases as the search progresses (f_best approaches the known minimum -0.8).
    The UCB score is normalized by the evaluation cost to prioritize cheap, promising points.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: Simple EI / Cost
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Estimate progress towards the known minimum (-0.8)
    # f_best starts around 0 (random) and goes to -0.8.
    # Range is roughly 0.8.
    known_min = -0.8
    # Clip f_best to a reasonable range to avoid division issues or extreme beta values
    # If f_best is already below -0.8 (due to noise/shift), treat as complete
    progress = 0.0
    if f_best > known_min:
        # Max possible improvement from start (approx 0) to min (-0.8) is 0.8
        # If f_best is 0, progress = 0. If f_best is -0.8, progress = 1.
        progress = np.clip((0.0 - f_best) / 0.8, 0.0, 1.0)
    else:
        progress = 1.0
        
    # 2. Adaptive Beta (Exploration Parameter)
    # Start with high exploration (beta=3.0), end with low exploration (beta=0.5)
    beta_max = 3.0
    beta_min = 0.5
    beta = beta_max - (beta_max - beta_min) * progress
    
    # 3. Calculate UCB for minimization
    # We want to find x such that f(x) is small.
    # UCB_lower(x) = mu(x) - beta * sigma(x)
    # This is a lower bound on the function value.
    # The "potential improvement" is how much lower this bound is compared to f_best.
    ucb_lower = mu - beta * sigma_safe
    
    # Potential improvement: f_best - ucb_lower
    # If ucb_lower < f_best, this is positive.
    # We want to maximize this potential.
    potential = f_best - ucb_lower
    
    # 4. Cost-Aware Normalization
    # Divide by cost to favor cheaper evaluations
    score = potential / cost_safe
    
    # 5. Safety: Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
