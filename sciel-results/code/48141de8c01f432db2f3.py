"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition blending Expected Improvement (EI) with
    probability of improvement (PI) and a small uncertainty bonus.
    
    The core signal is EI (captures expected reduction in objective value),
    augmented by PI (robust probability of beating f_best) and a mild
    exploration term for high-uncertainty regions. All are normalized by
    cost to favor cheap evaluations.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points (in objective units).
    f_best : float
        Best observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    start_time = time.time()
    deadline = 320  # safety margin from 330s limit
    
    # Ensure sigma is positive
    sigma_safe = np.clip(sigma, 1e-10, None)
    
    # Compute z-score: probability that f(x) < f_best
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Posterior probability of improvement (exceeding current best for minimization)
    p_improve = norm.cdf(z)
    p_improve = np.clip(p_improve, 1e-10, 1.0 - 1e-10)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where phi is the standard normal PDF
    phi_z = norm.pdf(z)
    ei = (f_best - mu) * p_improve + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Cost normalization: cost is in [1, 2]
    # Use inverse cost as a weight to favor cheaper evaluations
    cost_weight = 1.0 / np.maximum(cost, 1e-10)
    
    # Base score: blend EI and PI, weighted by cost efficiency
    # EI captures magnitude of expected improvement, PI captures confidence
    score = (0.6 * ei + 0.4 * p_improve) * cost_weight
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps navigate the Rosenbrock valley by encouraging exploration
    # when uncertainty is high relative to the scale of the problem
    # Scale uncertainty by the range of mu to make it adaptive
    mu_range = np.max(mu) - np.min(mu) + 1e-10
    uncertainty_bonus = 0.1 * (sigma_safe / mu_range)
    uncertainty_bonus = np.clip(uncertainty_bonus, 0, 1.0)
    
    # Blend the main score with uncertainty bonus
    score = 0.85 * score + 0.15 * uncertainty_bonus * cost_weight
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - start_time > deadline:
        pass  # The computation is already done, just return
    
    return score
# EVOLVE-BLOCK-END
