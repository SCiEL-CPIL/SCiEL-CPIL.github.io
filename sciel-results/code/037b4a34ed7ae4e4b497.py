"""Cost-aware acquisition for Ackley 2D BO with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # Safety margin before 300s limit

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid of Cost-Weighted Expected Improvement (EI/cost) and
    Uncertainty-weighted exploitation to balance exploration and cost efficiency.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Check deadline
    if _check_deadline():
        # Fallback: Exploitative score based on low mean, penalized by cost
        mu_safe = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        cost_safe = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)
        with np.errstate(divide='ignore', invalid='ignore'):
            fallback_scores = -mu_safe / (cost_safe + 1e-9)
        fallback_scores = np.nan_to_num(fallback_scores, nan=0.0, posinf=0.0, neginf=0.0)
        return fallback_scores
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace NaN/Inf with reasonable defaults
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)
    
    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)
    
    # Compute z-score: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -50, 50)
    
    # Expected Improvement (EI)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI score
    ei_cost = ei / cost
    
    # Uncertainty-weighted exploitation:
    # When sigma is high relative to the scale of the problem,
    # we want to explore. When sigma is low, we want to exploit low mu.
    # Use a combined score that blends exploitation (-mu/cost) with exploration (sigma/cost)
    # weighted by how confident we are.
    
    # Normalize mu to [0, 1] range for blending
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min + 1e-9
    mu_norm = (mu - mu_min) / mu_range
    
    # Normalize sigma to [0, 1] range
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min + 1e-9
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Exploitation score: prefer low mu, penalize cost
    exploit = (-mu / cost)
    
    # Exploration score: prefer high sigma, penalize cost
    explore = (sigma / cost)
    
    # Blend based on confidence: when sigma is high, explore more
    # When sigma is low, exploit more
    # Use a smooth blend weight
    blend_weight = 1.0 / (1.0 + sigma_norm * 5.0)
    
    # Combined score
    combined = blend_weight * exploit + (1.0 - blend_weight) * explore
    
    # Normalize both to comparable scales
    ei_cost_max = np.max(ei_cost) + 1e-9
    combined_max = np.max(np.abs(combined)) + 1e-9
    
    ei_cost_norm = ei_cost / ei_cost_max
    combined_norm = combined / combined_max
    
    # Final score: blend of EI/cost and the combined exploration/exploitation
    # Weight EI/cost more heavily as it's a proven acquisition function
    final_scores = 0.7 * ei_cost_norm + 0.3 * combined_norm
    
    # Replace any remaining NaN/Inf
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
