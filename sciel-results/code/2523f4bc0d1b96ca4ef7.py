import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Shekel 4d.
    
    Uses a hybrid approach:
    1. Compute EI and PoI
    2. Use adaptive blending based on uncertainty
    3. Apply cost-aware normalization with soft-min smoothing
    4. Add exploration bonus scaled by sigma
    """
    global _start_time
    
    # Initialize time tracking
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we've used most of the 330s budget, return quickly
    elapsed = time.time() - _start_time
    if elapsed > 320:  # 10s safety margin
        # Return a simple, fast score to avoid timeout
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        score = poi / np.maximum(cost, 1e-9)
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute EI: expected improvement over f_best
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute PoI: probability of improvement
    poi = norm.cdf(z)
    
    # Relative uncertainty: high when sigma is large relative to expected improvement
    rel_uncertainty = sigma_safe / (np.abs(f_best - mu) + sigma_safe + 1e-9)
    
    # Adaptive blending weight:
    # - When uncertainty is high (rel_uncertainty close to 1), favor PoI (exploration)
    # - When uncertainty is low (rel_uncertainty close to 0), favor EI (exploitation)
    # Use a smooth tanh-based transition
    alpha = 0.5 * (1 + np.tanh(4 * (rel_uncertainty - 0.5)))
    
    # Scale EI to be comparable to PoI (which is in [0,1])
    # Use a characteristic scale based on current uncertainty
    scale = np.maximum(np.abs(f_best - mu), sigma_safe)
    ei_scaled = ei / (scale + 1e-9)
    
    # Blend: alpha * EI_scaled + (1 - alpha) * PoI
    blended = alpha * ei_scaled + (1 - alpha) * poi
    
    # Add a small exploration bonus to encourage visiting uncertain regions
    # This helps escape local optima in multimodal Shekel 4d
    exploration_bonus = 0.1 * sigma_safe
    
    # Total acquisition score
    total_score = blended + exploration_bonus
    
    # Normalize by cost using a soft-min approach for smoother gradients
    # cost_safe = np.maximum(cost, 1e-9)
    # score = total_score / cost_safe
    
    # Use a smooth cost penalty: score = total_score * exp(-beta * (cost - 1))
    # This gives lower scores to expensive points but with smooth transitions
    beta = 0.5  # Cost sensitivity parameter
    cost_penalty = np.exp(-beta * (cost - 1.0))
    score = total_score * cost_penalty
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
