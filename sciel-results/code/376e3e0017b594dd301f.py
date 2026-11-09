"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290

    sigma_safe = np.maximum(sigma, 1e-9)

    # Hypervolume-inspired acquisition:
    # Estimate expected contribution to the Pareto front by combining:
    # 1. Expected improvement over f_best (in objective units)
    # 2. Uncertainty bonus (sigma) to explore regions with high variance
    # 3. Cost normalization to prefer cheaper evaluations
    
    # Expected improvement: E[max(f_best - f(x), 0)]
    # For Gaussian: EI = sigma * (phi(z) + z * Phi(z)) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    EI = sigma_safe * (phi_z + z * Phi_z)
    
    # Hypervolume-inspired score:
    # Weight expected improvement by a factor that rewards high uncertainty
    # (exploration) but penalizes points where improvement is unlikely.
    # The term (1 + sigma_safe) amplifies points with high uncertainty.
    # Divide by cost to normalize for evaluation expense.
    
    # Use a soft cost penalty: cost^0.7 instead of linear
    cost_penalty = np.power(cost, 0.7)
    
    # Hypervolume score: EI * uncertainty_bonus / cost_penalty
    # The uncertainty bonus (1 + sigma) encourages exploring uncertain regions
    # where the true function value could be much better than mu suggests.
    uncertainty_bonus = 1.0 + sigma_safe
    hv_score = EI * uncertainty_bonus / cost_penalty
    
    # Add a small exploration term based on pure uncertainty to ensure
    # we don't completely ignore high-sigma regions with moderate EI
    exploration_term = sigma_safe / cost_penalty
    
    # Combine: 80% hypervolume-inspired + 20% pure exploration
    score = 0.8 * hv_score + 0.2 * exploration_term
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return EI / cost_penalty
    
    return score
# EVOLVE-BLOCK-END
