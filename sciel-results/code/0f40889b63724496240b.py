"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_DEADLINE = None
def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        # 290s safety margin from 300s limit
        _DEADLINE = time.time() + 290
    return _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (POI)
    with dynamic uncertainty-based exploration.
    
    Parameters:
    - mu: (2000,) posterior mean in objective units
    - sigma: (2000,) posterior std in objective units  
    - f_best: scalar, lowest observed value
    - cost: (2000,) evaluation cost for each candidate
    
    Returns:
    - (2000,) scores, higher = evaluate next
    """
    t_start = time.time()
    deadline = _get_deadline()
    
    n = len(mu)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # 1. Probability of Improvement (POI)
    # P(f(x) < f_best) for f(x) ~ N(mu, sigma^2)
    # z = (f_best - mu) / sigma
    # POI = Phi(z)
    z = (f_best - mu) / (sigma + 1e-9)
    poi = norm.cdf(z)
    
    # 2. Dynamic Exploration Term
    # Normalize sigma by the maximum sigma in the current candidate set.
    # This makes the exploration term relative to the current state of knowledge.
    # If the model is very uncertain (high max sigma), the relative uncertainty 
    # of specific points is preserved. As the model learns (max sigma drops),
    # the exploration pressure naturally decays.
    max_sigma = np.max(sigma)
    if max_sigma < 1e-6:
        max_sigma = 1e-6
        
    # Relative uncertainty: sigma / max_sigma
    # We use this as a proxy for information gain or "surprise".
    rel_uncertainty = sigma / max_sigma
    
    # To avoid pure exploitation (POI is 0 or 1 in flat regions), 
    # we add a small exploration component.
    # The weight beta is tuned to balance POI and exploration.
    # Since POI is in [0, 1] and rel_uncertainty is in [0, 1], 
    # they are on similar scales.
    beta = 0.2
    
    # Combine POI and Exploration
    # Note: POI is the primary driver for finding better points.
    # Exploration helps escape local minima by visiting high-uncertainty areas.
    score_unnormalized = poi + beta * rel_uncertainty
    
    # 3. Cost Normalization
    # Higher cost should reduce the score.
    score = score_unnormalized / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() > deadline - 5:
        # Fallback: simple cost-normalized improvement potential
        # Use EI-like term for fallback to be safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        fallback = ei / cost
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    return score
# EVOLVE-BLOCK-END
