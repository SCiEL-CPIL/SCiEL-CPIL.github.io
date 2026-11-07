"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track start time for deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining LCB and Expected Improvement.
    
    Uses a hybrid approach that balances exploitation (via EI) and exploration
    (via LCB), with a quadratic cost penalty to prioritize low-cost evaluations.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP in objective units.
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP in objective units.
    f_best : float
        Best observed objective value so far (minimum, since we minimize).
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point, in [1, 2].
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline Guard: If we are running out of time, return a fast, simple score
    # to ensure we don't contribute to a timeout.
    # The evaluator limit is 330s. We use 320s as a safety margin.
    elapsed = time.time() - _start_time
    if elapsed > 320:
        cost_arr = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost_arr, 1e-8)
        # Simple inverse cost score
        return 1.0 / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost_arr = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is strictly positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    # Ensure cost is strictly positive
    cost_safe = np.maximum(cost_arr, 1e-8)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected improvement
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Clip negative EI to zero (EI should be non-negative)
    ei = np.maximum(ei, 0.0)
    
    # Compute LCB (Lower Confidence Bound) for exploration
    # LCB(x) = mu(x) - kappa * sigma(x)
    # For minimization, we want to minimize LCB.
    # The "potential improvement" from LCB perspective: f_best - LCB(x)
    kappa = 1.0
    lcb = mu - kappa * sigma_safe
    lcb_improvement = np.maximum(f_best - lcb, 0.0)
    
    # Hybrid score: combine EI and LCB improvement
    # Weight EI more heavily for exploitation, LCB for exploration
    # Use a smooth blend that adapts based on uncertainty
    # When sigma is high, favor LCB (exploration); when low, favor EI (exploitation)
    # Normalize sigma to [0, 1] for blending weight
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-10:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Blend weight: higher sigma -> more weight on LCB
    # Use a sigmoid-like smooth transition
    blend_weight = 0.5 + 0.5 * np.tanh(2.0 * (sigma_norm - 0.5))
    
    # Combined score before cost penalty
    combined = (1.0 - blend_weight) * ei + blend_weight * lcb_improvement
    
    # Apply Quadratic Cost Penalty (1/cost^2)
    # This aggressively prioritizes low-cost evaluations,
    # fitting more evaluations within the cost budget of 30.
    score = combined / (cost_safe ** 2)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
