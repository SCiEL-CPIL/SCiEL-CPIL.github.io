import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s to be safe within 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviation predictions.
        f_best: float, the lowest observed value.
        cost: (2000,) array of evaluation costs.
        
    Returns:
        (2000,) array of acquisition scores.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # If we are running out of time, return a safe default (e.g., uniform random or just 0)
        # But we must return finite scores.
        return np.zeros_like(mu)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in cost
    cost_safe = np.where(cost > 1e-6, cost, 1e-6)
    
    # Student-t variance factor scaling for sigma
    # Using nu=5 gives factor sqrt(5/3) ~ 1.29
    # This increases exploration in high-uncertainty regions
    nu = 5.0
    scale_factor = np.sqrt(nu / (nu - 2.0))
    sigma_eff = sigma * scale_factor
    
    # Calculate z
    # Add small epsilon to sigma to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma_eff, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: We use sigma_eff in z calculation, but standard EI formula uses sigma.
    # However, the "Student-t EI" or scaled EI typically scales the uncertainty.
    # If we scale sigma in z, we should also consider if the formula changes.
    # The finding says "sigma is scaled by the Student-t variance factor".
    # Let's stick to the standard EI formula but with scaled sigma in the denominator of z?
    # Actually, if we are approximating a t-distribution, the EI formula is different.
    # But the finding says "clean EI/cost ratio where sigma is scaled".
    # This implies: EI_scaled = (f_best - mu) * Phi(z_scaled) + sigma * phi(z_scaled) ??
    # Or EI_scaled = (f_best - mu) * Phi(z_scaled) + sigma_scaled * phi(z_scaled) ??
    # Usually, scaling sigma for exploration means treating the uncertainty as larger.
    # Let's use sigma_eff in both terms to be consistent with the "effective" uncertainty.
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_eff * phi_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    score = ei / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
