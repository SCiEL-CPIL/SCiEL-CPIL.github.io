"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf
import time

# Track global start time for deadline safety
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before 330s kill

# Constants for the cosine8 problem
MIN_POSSIBLE = -0.8
# Initial f_best is likely around -0.2 to 0.2
# We use a progress metric that goes from 1.0 (start) to 0.0 (optimum)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of cosine8 8d.
    
    Refinement of EI/UCB blend:
    - Uses robust normalization (median/MAD) to handle outliers in candidate scores.
    - Adaptive exploration weight based on progress towards the known minimum.
    - Cost adjustment via a smooth penalty function rather than simple division.
    """
    # Deadline guard
    if time.time() - _START_TIME > 330.0 - _DEADLINE_MARGIN:
        return np.zeros_like(mu)

    n = len(mu)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1.0)  # Cost is in [1, 2]

    # 1. Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -20, 20)
    
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    phi_z = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    # phi(z) = exp(-z^2 / 2) / sqrt(2 * pi)
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    ei = (f_best - mu) * phi_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)

    # 2. Uncertainty (UCB component)
    # We use sigma as the exploration term.
    ucb_term = sigma_safe

    # 3. Cost Adjustment
    # Instead of dividing by cost, which can amplify noise, we use a weighted average
    # or a penalty. Here, we scale the improvement by an inverse cost factor.
    # cost_safe is in [1, 2]. 
    # Let's use cost_factor = 1.0 / (cost_safe ** 0.5) to soften the penalty.
    cost_factor = 1.0 / np.sqrt(cost_safe)
    
    ei_cost = ei * cost_factor
    ucb_cost = ucb_term * cost_factor

    # 4. Robust Normalization
    # Using median and MAD (Median Absolute Deviation) is more robust to outliers
    # than mean and std, which can be skewed by a few extreme candidates.
    
    ei_med = np.median(ei_cost)
    ei_mad = np.median(np.abs(ei_cost - ei_med)) + 1e-12
    ei_norm = (ei_cost - ei_med) / ei_mad

    ucb_med = np.median(ucb_cost)
    ucb_mad = np.median(np.abs(ucb_cost - ucb_med)) + 1e-12
    ucb_norm = (ucb_cost - ucb_med) / ucb_mad

    # 5. Adaptive Blending
    # Progress: 0 when f_best is near -0.8, 1 when f_best is far (e.g., 0.0)
    # We assume initial f_best is around 0.0 to -0.2.
    # Let's define progress range from -0.8 (optimum) to 0.5 (worst expected start)
    progress = np.clip((f_best - MIN_POSSIBLE) / (0.5 - MIN_POSSIBLE), 0.0, 1.0)
    
    # Exploration weight: Higher when progress is high (far from optimum)
    # Use a sigmoidal curve for smoother transition
    # w_explore = 1 / (1 + exp(-k * (progress - 0.5)))
    # At progress=0.5, w=0.5. At progress=1.0, w~0.88. At progress=0.0, w~0.12.
    w_explore = 1.0 / (1.0 + np.exp(-5.0 * (progress - 0.5)))
    w_exploit = 1.0 - w_explore

    # 6. Combine
    score = w_exploit * ei_norm + w_explore * ucb_norm

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
