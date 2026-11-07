import numpy as np
import time
from scipy.stats import norm

# Global state for deadline guard and progress estimation
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _call_count
    
    # Initialize time on first call
    if _start_time is None:
        _start_time = time.time()
    
    _call_count += 1
    
    # Deadline guard: if we are close to the 330s limit, return safe exploitation scores
    # to avoid any potential overhead or instability near the cutoff.
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Simple exploitation: favor low mu, penalized by cost
        # f_best - mu is the improvement.
        scores = (f_best - mu) / np.maximum(cost, 1e-9)
        # Clip to ensure finite
        scores = np.clip(scores, -1e6, 1e6)
        return scores

    # Estimate progress based on call count
    # Total budget 30. Start cost ~7.5. Remaining 22.5.
    # Avg cost ~1.5 -> ~15 evals.
    # Total calls ~15-20.
    max_calls = 20
    progress = min(1.0, _call_count / max_calls)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    # Handle sigma == 0 case to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    Z = (f_best - mu) / sigma_safe
    
    # Phi and phi are CDF and PDF of standard normal
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # Calculate UCB score for minimization
    # Optimistic lower bound: mu - beta * sigma
    # Gain: f_best - (mu - beta * sigma) = f_best - mu + beta * sigma
    
    # Decay beta from 3.0 to 1.0
    beta_max = 3.0
    beta_min = 1.0
    beta = beta_max - (beta_max - beta_min) * progress
    
    UCB_score = f_best - mu + beta * sigma_safe
    
    # Mix EI and UCB
    # Early on, rely more on UCB for exploration.
    # Later, rely more on EI for exploitation.
    # w_ucb decays from 1.0 to 0.2
    # w_ei increases from 0.2 to 1.0
    w_ucb = 1.0 - 0.8 * progress
    w_ei = 0.2 + 0.8 * progress
    
    combined_score = w_ucb * UCB_score + w_ei * EI
    
    # Divide by cost to be cost-aware
    scores = combined_score / np.maximum(cost, 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
