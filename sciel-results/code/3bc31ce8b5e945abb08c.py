"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # wall-clock limit in seconds
_SAFETY_MARGIN = 15.0  # seconds before deadline to start returning


def _check_deadline():
    """Return True if we should stop and return current best."""
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function returning 2000 finite scores.
    
    Uses cost-normalized Expected Improvement (EI) blended with UCB exploration.
    EI captures the expected magnitude of improvement over f_best.
    UCB adds exploration in high-uncertainty regions.
    Both are divided by cost to favor cheaper evaluations.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # For f(x) ~ N(mu, sigma^2): EI = (f_best - mu)*Phi(z) + sigma*phi(z)
    # where phi is the standard normal PDF
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    ei = (f_best - mu) * cdf_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # UCB exploration term: beta * sigma
    # Use a moderate beta to balance exploration
    beta = 0.5
    ucb = -mu + beta * sigma
    
    # Blend EI and UCB: weight EI more heavily since we want improvement
    # Use 70% EI + 30% UCB for balanced exploitation/exploration
    acq = 0.7 * ei + 0.3 * ucb
    
    # Cost normalization: divide by cost to favor cheaper evaluations
    # Cost ranges from 1 to 2, so this gives a meaningful relative penalty
    acq = acq / cost
    
    # Ensure finite values
    acq = np.where(np.isfinite(acq), acq, 0.0)
    
    # Ensure non-negative (clip small negatives from numerical noise)
    acq = np.maximum(acq, 0.0)
    
    return acq
# EVOLVE-BLOCK-END
