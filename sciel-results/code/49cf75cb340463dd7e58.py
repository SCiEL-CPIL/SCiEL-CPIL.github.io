"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline tracking
_start_time = time.time()
_DEADLINE = 325.0  # 5 seconds safety margin before 330s kill


def _check_deadline():
    """Check if we're close to the deadline."""
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural blend of
    Expected Improvement (EI) and Upper Confidence Bound (UCB),
    with adaptive exploration/exploitation balance.
    
    Parameters:
    - mu: (2000,) array of GP mean predictions
    - sigma: (2000,) array of GP std predictions
    - f_best: scalar, best observed value so far
    - cost: (2000,) array of costs for each candidate
    
    Returns:
    - (2000,) array of acquisition scores (higher = better)
    """
    # Check deadline
    if _check_deadline():
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and cost is positive
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Standard Gaussian EI (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB: balance exploitation and exploration
    # Use a beta parameter that adapts based on sigma magnitude
    # Larger sigma -> more exploration; smaller sigma -> more exploitation
    # beta controls the weight of the exploration term
    # We use a fixed beta that works well across the optimization trajectory
    beta = 1.8
    
    # UCB for minimization: -mu + beta * sigma
    ucb = -mu + beta * sigma
    
    # Normalize EI and UCB to comparable scales
    # EI is always non-negative; UCB can be negative
    # Scale EI by its mean to make it comparable to UCB scale
    ei_mean = np.mean(ei)
    if ei_mean > 1e-10:
        ei_normalized = ei / ei_mean
    else:
        ei_normalized = ei
    
    # For UCB, center it so that its mean is around 0
    ucb_mean = np.mean(ucb)
    ucb_normalized = (ucb - ucb_mean)
    
    # Adaptive blend: use more EI when we have a good best value (late optimization)
    # and more UCB when we're still exploring (early optimization)
    # Heuristic: if f_best is close to zero (for functions with min=0), we're in late stage
    # Use a sigmoid-like transition based on f_best magnitude
    # For hartmann, min is -3.86, so we need a different scale
    # Use a relative measure: how much improvement is possible
    # Simpler: use a fixed blend that balances both
    # Weight: 0.5 for EI, 0.5 for UCB, but scale them properly
    
    # Actually, let's use a different structural approach:
    # Use a "knowledge gradient" style: focus on points that reduce uncertainty
    # while also having potential for improvement
    
    # Combined score: weighted sum of normalized EI and UCB
    # Use a slightly higher weight on EI for exploitation
    weight_ei = 0.6
    weight_ucb = 0.4
    
    combined = weight_ei * ei_normalized + weight_ucb * ucb_normalized
    
    # Cost-aware scaling: divide by cost^power
    # Use power 0.7 to be moderately cost-aware
    cost_power = 0.7
    scores = combined / (cost ** cost_power)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
