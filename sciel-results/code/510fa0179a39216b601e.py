"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

_EPS = 1e-12

def _ackley_2d(x):
    """Ackley 2D function (standard, minimum at 0,0)."""
    x1 = x[:, 0]
    x2 = x[:, 1]
    sum_sq = x1**2 + x2**2
    sum_cos = np.cos(2.0 * np.pi * x1) + np.cos(2.0 * np.pi * x2)
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * sum_sq)) - np.exp(0.5 * sum_cos) + np.e + 20.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Expected Improvement (CEI).
    
    This approach maximizes the expected improvement per unit of cost.
    It is a principled alternative to weighted blends of EI, PI, and UCB.
    
    mu, sigma: (2000,) arrays of GP mean and std in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) array of evaluation costs
    
    Returns: (2000,) finite scores, higher = prefer to evaluate next
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, _EPS)
    
    # Ensure cost is positive
    cost = np.maximum(cost, _EPS)

    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma
    
    # Use scipy.stats for numerical stability
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)

    # Cost-Adjusted Expected Improvement
    cei = ei / cost

    # Ensure all values are finite
    cei = np.where(np.isfinite(cei), cei, 0.0)

    return cei
# EVOLVE-BLOCK-END
