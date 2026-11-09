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
    
    # Compute z-score: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (probability that f(x) < f_best)
    p_improve = np.clip(Phi_z, 1e-10, 1.0 - 1e-10)
    
    # Cost normalization: cost is in [1, 2]
    # Use inverse cost as a weight, but blend with a softer penalty
    # score = base_score / (1.0 + cost) is a common cost-aware form
    cost_denom = 1.0 + cost
    
    # Base EI score normalized by cost
    ei_score = ei / cost_denom
    
    # PI score normalized by cost
    pi_score = p_improve / cost_denom
    
    # Uncertainty bonus: encourages exploration in high-sigma regions
    # Scale by inverse of |mu| to avoid dominating when mu is near zero
    # Clip to [0, 1] for stability
    uncertainty_bonus = 0.1 * (sigma_safe / np.maximum(np.abs(mu), 1e-10))
    uncertainty_bonus = np.clip(uncertainty_bonus, 0.0, 1.0)
    ub_score = uncertainty_bonus / cost_denom
    
    # Blend: EI is the primary signal, PI adds robustness, UB adds exploration
    # Weights: 0.7 EI + 0.2 PI + 0.1 UB
    score = 0.7 * ei_score + 0.2 * pi_score + 0.1 * ub_score
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - start_time > deadline:
        pass  # The computation is already done, just return
    
    return score
# EVOLVE-BLOCK-END
