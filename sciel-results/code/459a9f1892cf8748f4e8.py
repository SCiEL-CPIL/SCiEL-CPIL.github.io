"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    This function computes the expected improvement of each candidate over the
    current best observed value, weighted by the cost of evaluation.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - lowest observed function value
        cost: array of shape (2000,) - cost of evaluating each candidate
        
    Returns:
        array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a valid finite array to avoid invalidating the result
        return np.random.uniform(size=mu.shape)

    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f(x), 0)]
    # Let Z = (f(x) - f_best) / sigma
    # Then f(x) = mu + sigma * Z
    # Improvement I = f_best - f(x) = (f_best - mu) - sigma * Z
    # E[I] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # Handle the case where sigma is very small to avoid division by zero
    # If sigma is tiny, the distribution is a point mass.
    # If mu < f_best, EI is approximately f_best - mu.
    # If mu >= f_best, EI is approximately 0.
    
    # To be numerically stable, we compute the standardized improvement
    # Let d = f_best - mu
    # Let z = d / sigma
    # EI = d * Phi(z) + sigma * phi(z)
    
    d = f_best - mu
    z = d / sigma_safe
    
    # Compute Phi(z) and phi(z)
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    
    # To avoid underflow/overflow issues with very large/small z,
    # we can use the asymptotic expansions or just rely on scipy's stability.
    # scipy.stats.norm.cdf and pdf are generally stable.
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = d * Phi_z + sigma_safe * phi_z
    
    # EI is always non-negative.
    # We want to maximize EI per unit cost.
    # However, linear cost division can be harsh.
    # Let's use a cost-aware weighting.
    # The task says "cost-aware". A common approach is EI / cost.
    # Another is EI * exp(-lambda * cost) or similar.
    # Given the cost range [1, 2], 1/cost is a reasonable baseline.
    # Let's stick with EI / cost for simplicity and direct interpretation,
    # but ensure it's finite.
    
    cost_safe = np.maximum(cost, 1e-8)
    scores = ei / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: Add a small exploration bonus for candidates with high sigma
    # even if EI is low, to prevent premature convergence?
    # EI already incorporates sigma. High sigma increases EI if mu is close to f_best.
    # If mu is much worse than f_best, EI is low even with high sigma.
    # This is correct behavior for EI.
    
    return scores
# EVOLVE-BLOCK-END
