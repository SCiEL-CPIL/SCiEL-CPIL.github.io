import numpy as np
from scipy.stats import t
import time

# Global state for deadline management
# The evaluator runs the whole script within a time limit.
# We track time from module import.
_START_TIME = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Student-t Expected Improvement (SEI) acquisition function.
    
    This function uses a Student-t distribution to model the predictive uncertainty,
    which provides heavier tails than the standard Gaussian assumption. This can
    improve exploration in multimodal landscapes like Rastrigin or Ackley.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: scalar - best observed value so far
        cost: array of shape (2000,) - evaluation costs
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard: If we are approaching the time limit, return a simple, fast heuristic
    # to ensure we don't crash the evaluator.
    if time.time() - _START_TIME > _TIME_LIMIT:
        cost_safe = np.maximum(cost, 1e-9)
        # Simple exploitation: prioritize low mu, penalized by cost
        return -mu / cost_safe

    # Numerical stability for sigma
    sig = np.maximum(sigma, 1e-9)
    
    # Standardized distance
    z = (f_best - mu) / sig
    
    # Clip z to avoid extreme values that might cause numerical issues in t-distribution functions
    # t-distribution is well-defined for all real z, but CDF/PDF underflow/overflow can occur
    z = np.clip(z, -100.0, 100.0)
    
    # Degrees of freedom for Student-t
    # Nu=3 provides heavier tails, better for multimodal landscapes like Rastrigin
    nu = 3.0
    
    # CDF of t-distribution with nu dof
    # F_nu(z)
    F_nu = t.cdf(z, nu)
    
    # PDF of t-distribution with nu-1 dof
    # pdf_nu_minus_1(z)
    pdf_nu_minus_1 = t.pdf(z, nu - 1)
    
    # Pre-compute constant factor
    factor = (nu - 1) / (nu - 2)
    
    # Student-t Expected Improvement
    # Formula: EI = sigma * [ factor * pdf_{nu-1}(z) + z * F_nu(z) ]
    ei_t = sig * (factor * pdf_nu_minus_1 + z * F_nu)
    
    # Ensure non-negative (numerical safety)
    ei_t = np.maximum(ei_t, 0.0)
    
    # Cost-aware score: Higher EI per unit cost is better
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei_t / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
