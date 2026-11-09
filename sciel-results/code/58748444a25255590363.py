"""
Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Uses a structural blend of Expected Improvement (EI) and Uncertainty (UCB)
normalized by cost to balance exploration and exploitation in a multimodal
landscape under a cost constraint.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import erfc

# Deadline guard constants
_START_TIME = time.time()
_DEADLINE_SECONDS = 290.0

def _check_deadline():
    """Check if we are close to the time limit."""
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        return True
    return False

def _fast_ei(mu, sigma, f_best):
    """
    Fast calculation of Expected Improvement.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) = 0.5 * erfc(-z / sqrt(2))
    phi_z = 0.5 * erfc(-z / np.sqrt(2.0))
    
    # pdf(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    pdf_z = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z)
    
    ei = (f_best - mu) * phi_z + sigma_safe * pdf_z
    
    # Ensure non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    Structural change: Uses a blend of Expected Improvement (EI) and
    Uncertainty (sigma) weighted by an adaptive beta, normalized by cost.
    This directly optimizes expected improvement per unit cost while
    maintaining exploration in high-uncertainty regions typical of
    multimodal functions like Rastrigin.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check deadline
    if _check_deadline():
        # Fallback: simple cost-aware EI
        cost_safe = np.maximum(cost, 1e-9)
        ei = _fast_ei(mu, sigma, f_best)
        scores = ei / cost_safe
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Calculate Expected Improvement
    ei_score = _fast_ei(mu, sigma, f_best)
    
    # 2. Determine adaptive beta for UCB component
    # Rastrigin is multimodal. Early on, we need strong exploration.
    # As f_best approaches 0, we switch to exploitation.
    if f_best > 10.0:
        beta = 2.5
    elif f_best > 5.0:
        beta = 1.5
    elif f_best > 2.0:
        beta = 0.8
    elif f_best > 0.5:
        beta = 0.4
    else:
        beta = 0.1
        
    # 3. Combine EI and UCB (sigma)
    # The UCB term is beta * sigma.
    # We add it to EI to encourage exploration of uncertain regions.
    combined_score = ei_score + beta * sigma
    
    # 4. Cost-aware normalization
    # Divide by cost to prefer cheaper evaluations.
    # Cost is in [1, 2].
    scores = combined_score / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
