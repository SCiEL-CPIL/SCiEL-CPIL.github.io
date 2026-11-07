"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 3D.
    
    Structural change: Use a pure Expected Improvement (EI) acquisition function
    with a small, fixed cost penalty. This is a deterministic approach that
    balances exploitation (via EI) with cost-awareness (via the penalty term).
    
    The EI naturally handles exploration through uncertainty, and the cost
    penalty gently steers the search toward cheaper regions.
    
    Key insight: The minimum is uniformly distributed in the inner 80% of the
    domain, so we should not aggressively penalize cost. A small lambda (0.1)
    is sufficient to guide the search without dominating the EI signal.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill
    
    # Ensure inputs are valid
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    n = len(mu)
    
    # Basic safety: if anything is wrong, return a simple fallback
    if n == 0 or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.zeros(n, dtype=np.float64)
    
    # Clamp sigma to positive values for numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Numerical stability
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Cost penalty: the cost is 1 + mean(x_scaled), so higher cost = more expensive
    # We want to penalize expensive points. The cost ranges from ~1.0 to ~2.0
    # Use a small lambda to gently steer toward cheaper regions
    lambda_cost = 0.1
    cost_penalty = lambda_cost * (cost - 1.0)
    
    # Combined score: EI - cost_penalty
    # Higher score means more attractive to evaluate
    score = ei - cost_penalty
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)
    
    # Check time
    if time.time() > deadline - 2.0:
        # Fallback to simple EI if running out of time
        score = ei
        score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
