"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the 330s limit, return safe default
    elapsed = _get_elapsed()
    if elapsed > 320:
        return np.zeros(len(mu), dtype=np.float64)

    # Cost-aware Adaptive UCB acquisition function
    # Structural change: use Expected Improvement (EI) combined with exploration bonus
    # normalized by cost, rather than Thompson Sampling
    
    n = len(mu)
    
    # Clip sigma to avoid numerical issues
    sigma_clipped = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    delta = f_best - mu
    z = delta / sigma_clipped
    
    # Avoid division by zero and extreme values
    z = np.clip(z, -50, 50)
    
    ei = delta * norm.cdf(z) + sigma_clipped * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Exploration bonus: weighted by uncertainty
    # Use a fixed exploration parameter that balances exploitation and exploration
    xi = 0.1  # exploration-exploitation trade-off parameter
    
    # Combine EI with exploration bonus, normalized by cost
    # The exploration bonus encourages visiting high-uncertainty regions
    exploration_bonus = xi * sigma_clipped
    
    # Total benefit: EI + exploration bonus
    benefit = ei + exploration_bonus
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    # Cost is in range [1, 2] based on the task description
    scores = benefit / (cost + 1e-8)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
