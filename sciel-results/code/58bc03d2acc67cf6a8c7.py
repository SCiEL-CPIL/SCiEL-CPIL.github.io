"""Rewritten cost-aware acquisition for shekel 4d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s. We return best result before that.
_DEADLINE_SECONDS = 325.0
_start_time = None

def _check_deadline():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > _DEADLINE_SECONDS


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining:
    - Expected Improvement per unit cost (CEA)
    - UCB-style exploration term (linear in cost)
    
    mu: (2000,) GP mean in objective units
    sigma: (2000,) GP std in objective units
    f_best: float, lowest observed value
    cost: (2000,) evaluation cost for each candidate (1 + mean(x_scaled_to_01))
    
    Returns: (2000,) finite scores, higher = evaluate next
    """
    # Safety: ensure inputs are finite and properly shaped
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # --- Component 1: Expected Improvement per unit cost (CEA) ---
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    cea = ei / cost
    
    # --- Component 2: UCB with linear cost penalty ---
    # beta controls exploration; we use a moderate value
    beta = 2.0
    ucb = mu + beta * sigma_safe
    # Linear cost penalty: subtract cost to discourage expensive points
    ucb_penalized = ucb - cost
    
    # --- Hybrid: combine CEA and penalized UCB ---
    # Weight them; CEA handles exploitation, UCB handles exploration
    # Use a fixed blend; both are in "score" units where higher is better
    # Normalize roughly by their scales
    w_cea = 0.5
    w_ucb = 0.5
    score = w_cea * cea + w_ucb * ucb_penalized
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
