import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Strategy:
    - Sample from GP posterior: y = mu - sigma * z where z ~ N(0,1)
    - Select candidates that minimize y / cost (best improvement per unit cost)
    - Thompson Sampling naturally balances exploration/exploitation without
      tuning hyperparameters like beta or alpha
    - Direct cost division in the sampled objective provides cost-awareness
    """
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 290.0  # 10s safety margin before 300s limit

    # Ensure inputs are 1D arrays
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    cost = np.asarray(cost).flatten()
    
    n = len(mu)
    
    # Guard against zero/negative sigma and cost
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Thompson Sampling: draw one sample from GP posterior
    # For minimization, we want low values, so we sample y = mu - sigma * z
    # where z ~ N(0,1). This gives us a random draw from the posterior.
    z = np.random.standard_normal(n)
    y_sampled = mu - sigma_safe * z
    
    # Score: minimize y_sampled / cost (lower is better)
    # But acquisition function should return HIGHER = evaluate next
    # So we negate: higher score = better candidate
    scores = -y_sampled / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: return simple EI/cost if we are extremely close to timeout
        z_det = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z_det) + sigma_safe * norm.pdf(z_det)
        ei = np.maximum(ei, 0.0)
        fallback_scores = ei / cost_safe
        return np.nan_to_num(fallback_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
