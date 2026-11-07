import numpy as np
from scipy.stats import norm
import time

# ---------- Global timer / safety ----------
_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin below 330s wall-clock

# ---------- Acquisition Parameters ----------
# Thompson Sampling with cost-aware weighting
# We draw a single posterior sample and score by -sample / cost^alpha
_COST_POW = 1.0  # Linear cost scaling for Thompson Sampling

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    Returns scores of shape (2000,), higher = evaluate next.
    
    Strategy: Draw a single sample from the posterior Gaussian N(mu, sigma)
    at each candidate point. Score candidates by the ratio of the negative
    sampled value to the evaluation cost. This stochastically balances
    exploration and exploitation without relying on deterministic confidence
    bounds or expected improvement calculations.
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: return EI / Cost
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, -1e9)
        return score

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: draw a single sample from the posterior
    # y = mu + sigma * z, where z ~ N(0, 1)
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility within a call
    z_sample = rng.standard_normal(mu.shape)
    y_sample = mu + sigma_safe * z_sample
    
    # Score: we want to minimize, so lower y_sample is better
    # Cost-aware: divide by cost to penalize expensive evaluations
    # Higher score = more likely to be evaluated next
    # We want to select candidates with low y_sample and low cost
    # So score = -y_sample / cost^alpha
    cost_penalty = cost_safe ** _COST_POW
    score = -y_sample / cost_penalty
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -1e9)
    
    return score
