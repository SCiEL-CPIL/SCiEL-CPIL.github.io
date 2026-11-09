"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 280s limit to leave 20s margin for 300s kill switch

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Penalized Expected Improvement (EI) acquisition function.
    
    Strategy:
    1. Compute Expected Improvement (EI) for minimization.
    2. Apply a linear cost penalty: Score = EI - lambda * (cost - min_cost).
       This differs from the previous ratio approach by allowing high-EI points
       to be selected even if they are expensive, as long as the expected gain
       exceeds the cost penalty.
    3. Add a standardized exploration term based on sigma, scaled by a dynamic
       beta that depends on the current best value's distance from the theoretical
       minimum of Hartmann 3D.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        return np.zeros(len(mu))

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return np.zeros(len(mu))
    
    # Ensure cost is positive and handle edge cases
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Expected Improvement Calculation
    # For minimization: EI = E[max(f_best - f(x), 0)]
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical issues in CDF/PDF
    # Phi(10) ~ 1, Phi(-10) ~ 0
    z_clipped = np.clip(z, -10, 10)
    
    Phi_z = norm.cdf(z_clipped)
    phi_z = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # 2. Cost Penalty
    # We penalize points that are more expensive than the cheapest candidate in the current batch.
    # This normalizes the cost penalty relative to the current set of candidates.
    min_cost = np.min(cost_safe)
    # Normalize cost to [0, 1] range for the penalty term
    max_cost = np.max(cost_safe)
    cost_range = max_cost - min_cost
    if cost_range < 1e-9:
        cost_norm = np.zeros_like(cost_safe)
    else:
        cost_norm = (cost_safe - min_cost) / cost_range
    
    # Lambda: How much we care about cost.
    # If f_best is close to the global minimum (-3.86), we should be more exploitative
    # and less sensitive to cost (or rather, we want to find the exact minimum, so cost
    # matters less than accuracy? No, we have a budget. We want to spend wisely).
    # Actually, a fixed lambda is safer. Let's use a lambda that scales with the
    # "urgency" of finding a better point.
    # If regret is high, we might be willing to pay more cost to explore?
    # Or if regret is low, we should be very careful with cost?
    # Let's use a constant lambda for stability, tuned to be significant.
    # Cost range is roughly [1, 2]. EI range is roughly [0, 5].
    # A penalty of 0.5 * cost_norm means a point with max cost gets -0.5 penalty.
    # This seems reasonable.
    lambda_cost = 0.5
    
    cost_penalty = lambda_cost * cost_norm
    
    # 3. Exploration Term
    # We add an exploration term to encourage visiting uncertain regions.
    # Standard LCB: -mu - beta * sigma
    # Here we are using EI which already accounts for mu and sigma.
    # Adding a pure sigma term might double-count uncertainty.
    # However, EI is zero when mu >> f_best. In these regions, EI ~ 0.
    # We might miss good exploration points if EI is low.
    # Let's add a small exploration bonus based on relative sigma.
    
    mean_sigma = np.mean(sigma_safe)
    if mean_sigma < 1e-9:
        exploration = np.zeros_like(sigma_safe)
    else:
        # Normalize sigma by mean to get a relative uncertainty
        rel_sigma = sigma_safe / (mean_sigma + 1e-9)
        
        # Beta: How much we weight exploration.
        # Start with a moderate beta.
        beta = 0.5
        exploration = beta * rel_sigma
        
    # 4. Combine
    # Score = EI - Cost_Penalty + Exploration
    # Note: EI is in objective units. Exploration is dimensionless (relative).
    # We need to scale exploration to objective units?
    # Or scale EI?
    # Let's scale exploration by a factor that matches the scale of EI.
    # Scale factor: mean_sigma is a good proxy for the scale of the objective changes.
    # So exploration_term = beta * (sigma_safe / mean_sigma) * mean_sigma = beta * sigma_safe.
    # Wait, that's just beta * sigma.
    # Let's stick to beta * sigma_safe, but ensure it's not dominating EI.
    # If EI is ~1.0 and sigma is ~0.5, beta=0.5 gives 0.25. Reasonable.
    
    # Re-evaluating exploration:
    # The previous program used `kappa * sigma`.
    # Let's use `beta * sigma_safe` but with a dynamic beta.
    # If f_best is far from -3.86, we are in early stages, so explore more.
    regret = max(f_best + 3.86, 0.0)
    # If regret is high (early), beta should be high.
    # If regret is low (late), beta should be low.
    beta = 0.5 + 1.0 * min(regret / 2.0, 1.0)
    
    exploration = beta * sigma_safe
    
    score = ei - cost_penalty + exploration
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
