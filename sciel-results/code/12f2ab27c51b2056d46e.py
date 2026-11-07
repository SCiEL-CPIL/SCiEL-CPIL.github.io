import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Strategy:
    - Samples a single function value from the GP posterior for each candidate.
    - Selects the point that minimizes the ratio of the sampled value to its cost.
    - This stochastic approach balances exploration and exploitation without
      tuning hyperparameters.
    """
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 290.0  # 10s safety margin before 300s limit

    # Ensure inputs are 1D arrays
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    cost = np.asarray(cost).flatten()
    
    # Guard against zero/negative sigma and cost
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Thompson Sampling:
    # 1. Sample from the posterior: f ~ N(mu, sigma^2)
    #    f_sampled = mu + sigma * z, where z ~ N(0, 1)
    z = np.random.standard_normal(len(mu))
    f_sampled = mu + sigma_safe * z
    
    # 2. Minimize the ratio of sampled value to cost
    #    We want to minimize R = f_sampled / cost
    #    Since the return value is "higher = evaluate next", we return -R
    scores = -f_sampled / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: return simple deterministic EI/cost if we are extremely close to timeout
        # This is a very fast fallback
        z_det = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * 0.5 * (1 + np.tanh(z_det)) # Approximate CDF with tanh for speed? No, just use simple UCB
        # Actually, just return -mu/cost as a simple exploit fallback
        fallback_scores = -mu / cost_safe
        return np.nan_to_num(fallback_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
