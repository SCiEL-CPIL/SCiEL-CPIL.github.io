<<<<<<< SEARCH
import numpy as np
import time

# Module-level time tracking for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition for Bayesian optimization.
    
    Uses a linear UCB heuristic: (f_best - mu) + kappa * sigma, divided by cost.
    This is a simpler, faster alternative to Expected Improvement that balances
    exploitation (low mu) and exploration (high sigma) more aggressively.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return quickly if we're getting close to the limit
    if time.time() - _start_time > 325:
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    N = mu.shape[0]
    
    # Ensure sigma is non-negative (it's a standard deviation)
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-10)
    
    # UCB for minimization: we want to minimize, so the "upper bound" on the
    # negative improvement is (f_best - mu) + kappa * sigma
    # Higher values of this quantity indicate more promising points.
    
    # Adaptive kappa: start with a moderate exploration bonus
    # The scale of sigma relative to f_best matters. Since targets are standardized
    # in the GP but mu/sigma are in objective units, we need a reasonable kappa.
    # For Griewank, values range from 0 to ~0.5 in the relevant region.
    # A kappa of 1.0-2.0 is reasonable for a UCB-style approach.
    
    # Use a slightly aggressive exploration factor
    kappa = 1.5
    
    # UCB score: (f_best - mu) + kappa * sigma
    # This is positive when mu < f_best (improvement expected) or when sigma is large
    ucb = (f_best - mu) + kappa * sigma_safe
    
    # Clip to be non-negative to avoid negative scores (which would mean "avoid this point")
    # Actually, we should allow negative scores since they indicate points worse than f_best
    # with low uncertainty. But for cost-aware scoring, we want higher = better.
    # Points with negative UCB are less attractive.
    
    # Cost-aware score: UCB / cost
    # Higher score means better expected improvement per unit cost
    score = ucb / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
=======
import numpy as np
import time
from scipy.special import erf, exp

# Module-level time tracking for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition for Bayesian optimization.
    
    Uses the standard Expected Improvement formula for minimization,
    divided by the evaluation cost. This provides a probabilistic measure
    of improvement per unit cost.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return quickly if we're getting close to the limit
    if time.time() - _start_time > 325:
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-10)
    
    # Calculate Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # z-score
    z = (f_best - mu) / sigma_safe
    
    # CDF of standard normal: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    sqrt_2 = np.sqrt(2.0)
    cdf_vals = 0.5 * (1.0 + erf(z / sqrt_2))
    
    # PDF of standard normal: phi(z) = exp(-z^2 / 2) / sqrt(2*pi)
    pdf_vals = exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    score = ei / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
>>>>>>> REPLACE
