"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with a small linear cost penalty.
    
    The cost of evaluating a point is 1 + mean(x) where x is in [0,1]^3.
    Since we don't have x directly, we use the provided cost array.
    
    We use a deterministic cost-aware weighting that balances exploitation (high EI)
    and cost-efficiency. The acquisition score is:
    
    score = EI - lambda * cost
    
    where lambda is tuned to be small (0.1) to gently prefer cheaper evaluations
    without overly biasing the search away from potentially optimal but expensive regions.
    This approach provides a principled, deterministic balance compared to
    stochastic UCB or aggressive multiplicative penalties.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # Since we are minimizing, improvement is f_best - mu (positive if mu < f_best)
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We compute this in a numerically stable way.
    
    # For large positive z, Phi(z) -> 1, phi(z) -> 0, EI -> f_best - mu
    # For large negative z, Phi(z) -> 0, phi(z) -> 0, EI -> 0
    
    # Use scipy's norm for CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = diff * cdf_z + sigma * pdf_z
    
    # Ensure finiteness and non-negativity (EI should be >= 0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware weighting: subtract lambda * cost
    # cost is in [1, 2] for x in [0,1]^3
    # Use lambda=0.1 for a gentle penalty
    lambda_cost = 0.1
    cost_safe = np.maximum(cost, 1e-9)
    
    # Acquisition score
    score = ei - lambda_cost * cost_safe
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
