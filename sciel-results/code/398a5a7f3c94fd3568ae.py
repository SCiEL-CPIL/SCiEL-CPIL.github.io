"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (Hartmann 3D).
    
    Uses Expected Improvement (EI) for minimization with a mild exploration
    boost and a small cost penalty to prefer cheaper evaluations when
    acquisition values are comparable.
    
    The deadline guard ensures we return within the 330s wall-clock limit.
    """
    start_time = time.time()
    deadline = start_time + 325.0  # 5s safety margin before 330s limit
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    n = mu.shape[0]
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is non-negative and zero when mu >= f_best with small sigma
    # Add a small exploration term to encourage visiting uncertain regions
    # but keep it modest so exploitation dominates when EI is clear
    exploration_weight = 0.3
    exploration = exploration_weight * sigma_safe
    
    # Cost penalty: cost is in [1, 2] (1 + mean(x) where x in [0,1]^3)
    # Normalize to [0, 1] and apply a small penalty
    cost_norm = np.clip((cost - 1.0), 0.0, 1.0)
    cost_penalty = 0.05 * cost_norm
    
    # Combine: higher score = better candidate to evaluate next
    score = ei + exploration - cost_penalty
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check deadline; if close, return immediately
    if time.time() > deadline - 1.0:
        return score
    
    # Return 2000 finite scores (higher = evaluate next)
    return score
# EVOLVE-BLOCK-END
