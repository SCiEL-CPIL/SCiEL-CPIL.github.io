"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: total wall-clock budget is 330s, leave 30s margin
_DEADLINE = time.time() + 300.0

def _check_deadline():
    return time.time() < _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a modified Expected Improvement (EI) that accounts for:
    1. Cost of evaluation (higher cost = lower priority)
    2. Uncertainty (higher sigma = more exploration)
    3. Current best (f_best)
    
    We use a combination of:
    - Standard EI for exploitation
    - Uncertainty bonus for exploration
    - Cost normalization
    
    The key insight: we want to maximize expected improvement per unit cost,
    but also balance exploration vs exploitation.
    """
    # Ensure sigma is positive and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    # But also add an uncertainty component to encourage exploration
    # when the GP is uncertain
    
    # Use a weighted combination:
    # - 70% cost-normalized EI (exploitation)
    # - 30% uncertainty/cost (exploration)
    # This helps avoid getting stuck in local optima
    
    cost_safe = np.maximum(cost, 1e-8)
    
    # Exploration term: normalized uncertainty
    # Scale by the typical EI magnitude to make it comparable
    ei_scale = np.maximum(np.abs(f_best - mu), 1e-8)
    exploration = sigma_safe / (cost_safe * (ei_scale + 1e-8))
    
    # Combine: cost-normalized EI + exploration bonus
    # The exploration term is scaled to be meaningful relative to EI
    score = (ei / cost_safe) + 0.5 * exploration
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
