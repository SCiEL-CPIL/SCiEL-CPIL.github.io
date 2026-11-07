"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = _start_time + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a hybrid of Expected Improvement and
    posterior sampling, with adaptive cost weighting.
    
    Strategy:
    1. Compute EI as baseline exploitation signal
    2. Add a stochastic exploration term scaled by uncertainty and inverse cost
    3. Use a cost-dependent exploration coefficient that shifts from 
       exploration to exploitation as budget is consumed
    """
    # Handle zero sigma
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Standard EI calculation
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Cost normalization: cost ranges from 1 to 2 (since mean(x_scaled) in [0,1])
    # Lower cost = cheaper point, so we want to favor lower cost
    cost_norm = (cost - 1.0) / 1.0  # Normalize to [0, 1]
    
    # Adaptive exploration coefficient:
    # Early: high exploration (beta ~ 2.0), Late: low exploration (beta ~ 0.5)
    # Use f_best as a proxy for progress: as f_best decreases, we're more confident
    # For Rosenbrock, f_best starts high (~100+) and approaches 0
    progress = np.clip(1.0 - f_best / 50.0, 0.0, 1.0)
    beta = 2.0 - 1.5 * progress  # beta in [0.5, 2.0]
    
    # Cost-aware score:
    # Combine EI (exploitation) with uncertainty (exploration) weighted by cost
    # The cost penalty is applied multiplicatively to both terms
    cost_factor = 1.0 / (1.0 + cost_norm)  # Higher for cheaper points
    
    # Exploration term: scaled by beta and cost factor
    exploration = beta * sigma_safe * cost_factor
    
    # Exploitation term: EI scaled by cost factor
    exploitation = ei * cost_factor
    
    # Combined score
    score = exploitation + exploration
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
