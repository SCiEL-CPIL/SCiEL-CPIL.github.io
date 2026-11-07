import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track elapsed time and return early if we're getting close to the limit.
_DEADLINE = 320.0  # 10s safety margin
_start_time = [time.time()]

def _elapsed():
    return time.time() - _start_time[0]

def _reset_deadline():
    _start_time[0] = time.time()

def _maybe_reset_deadline():
    # If elapsed time is very small (first call), reset to be safe
    if _elapsed() < 0.01:
        _start_time[0] = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: (2000,) array of GP mean predictions
    - sigma: (2000,) array of GP standard deviations
    - f_best: float, lowest observed value
    - cost: (2000,) array of evaluation costs
    
    Returns:
    - scores: (2000,) array of acquisition scores (higher = more attractive)
    """
    _maybe_reset_deadline()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Structural change: Use a different acquisition strategy.
    # Instead of pure EI / cost^2, combine:
    # 1. Lower Confidence Bound (LCB) for exploration
    # 2. Cost-aware weighting with a softer penalty
    
    # LCB: mu - kappa * sigma (for minimization, lower is better)
    # We want to find low values, so we use:
    # lcb = mu - kappa * sigma
    # Higher acquisition score means we prefer lower lcb values.
    # So score_lcb = -lcb = -mu + kappa * sigma
    
    # Let's use a blend of EI and LCB, with cost awareness.
    
    # Cost normalization: cost ranges from 1.0 to ~1.333 (for 3D) or 1.5 (for 2D)
    # Normalize cost to [0, 1] range for better scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
    
    # Cost weight: prefer lower cost, but not too aggressively
    # Use 1 / (1 + cost_norm * alpha) with alpha controlling strength
    # alpha = 2.0 gives moderate cost preference
    alpha = 2.0
    cost_weight = 1.0 / (1.0 + cost_norm * alpha)
    
    # LCB with kappa
    # Choose kappa based on exploration-exploitation balance
    # kappa = 1.5 is a reasonable starting point
    kappa = 1.5
    lcb_score = -mu + kappa * sigma
    
    # Normalize EI and LCB to comparable scales
    # EI is in objective units, LCB is also in objective units
    # But their scales may differ, so normalize each
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    lcb_max = np.max(lcb_score)
    lcb_min = np.min(lcb_score)
    if lcb_max > lcb_min:
        lcb_norm = (lcb_score - lcb_min) / (lcb_max - lcb_min)
    else:
        lcb_norm = np.zeros_like(lcb_score)
    
    # Blend EI and LCB
    # Weight: 0.6 for EI, 0.4 for LCB
    w_ei = 0.6
    w_lcb = 0.4
    combined = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Apply cost weight
    scores = combined * cost_weight
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline
    if _elapsed() > _DEADLINE:
        return scores
    
    return scores

# Keep the original function name as the main entry point
# The above acquisition function is the main one.
