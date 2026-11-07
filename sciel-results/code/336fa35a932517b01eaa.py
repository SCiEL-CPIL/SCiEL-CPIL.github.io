"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 300.0  # seconds, safety margin before 330s kill
_TIMEOUT_MARGIN = 25.0  # return at least 25s before deadline

def _check_timeout():
    return (time.time() - _START_TIME) > (_DEADLINE - _TIMEOUT_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (Rastrigin 2D).
    
    Strategy: Hybrid LCB and Expected Improvement (EI) with adaptive exploration
    and cost-awareness.
    
    For multimodal functions like Rastrigin, pure EI can be overly conservative
    in high-uncertainty regions, while pure LCB can be too aggressive. We blend
    them:
    - LCB (Lower Confidence Bound) provides an optimistic lower bound that
      encourages exploration of high-variance regions.
    - EI provides a robust expected improvement metric that accounts for
      the probability and magnitude of improvement.
    - The blend weight adapts: more LCB early (exploration), more EI late (exploitation).
    
    Additionally, we apply a cost-aware scaling to prefer cheaper evaluations.
    """
    if _check_timeout():
        # Fallback: simple cost-aware exploitation
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases: non-positive sigma and cost
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    cost_safe = np.where(cost > 1e-8, cost, 1e-8)
    
    # Compute z-score
    diff = f_best - mu
    z = diff / sigma_safe
    z = np.clip(z, -10, 10)  # Clip for numerical stability
    
    # Expected Improvement (EI)
    ei = diff * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Adaptive LCB exploration coefficient
    # Use mean cost as a proxy for "budget spent".
    # Early (low cost): favor LCB (exploration, aggressive).
    # Late (high cost): favor EI (exploitation, magnitude of improvement).
    # Cost ranges from ~5 (start) to 30 (end).
    # Let's map cost to a weight for EI: w_ei = cost / 35
    # At cost=5: w_ei = 0.14 -> 14% EI, 86% LCB
    # At cost=30: w_ei = 0.86 -> 86% EI, 14% LCB
    # This ensures a smooth transition.
    
    mean_cost = np.mean(cost)
    w_ei = np.clip(mean_cost / 35.0, 0.1, 0.9)
    w_lcb = 1.0 - w_ei
    
    # LCB exploration coefficient: start high, decay as cost increases
    # beta = 2.5 - 0.1 * mean_cost
    # At cost=5: beta = 2.0
    # At cost=30: beta = 0.5
    beta = 2.5 - 0.1 * mean_cost
    beta = np.clip(beta, 0.5, 3.0)
    
    # Lower Confidence Bound (for minimization)
    lcb = mu - beta * sigma_safe
    
    # Potential improvement from LCB: how much better could the LCB be than f_best?
    lcb_improvement = np.maximum(0.0, f_best - lcb)
    
    # Normalize LCB improvement and EI to comparable scales
    # LCB improvement can be large if sigma is large, similar to EI
    # But we want to blend them smoothly
    max_lcb_imp = np.max(lcb_improvement)
    if max_lcb_imp < 1e-9:
        max_lcb_imp = 1.0
    
    # Blend: weighted sum of normalized LCB improvement and EI
    # Scale LCB improvement to be comparable to EI
    lcb_scaled = lcb_improvement / max_lcb_imp * np.max(ei) if np.max(ei) > 1e-9 else lcb_improvement
    
    blended_score = w_lcb * lcb_scaled + w_ei * ei
    
    # Cost-aware scaling
    # We want to prefer lower cost.
    score = blended_score / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
