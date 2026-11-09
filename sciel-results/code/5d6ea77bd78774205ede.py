# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for minimizing hartmann 3d.
    
    Uses a Lower Confidence Bound (LCB) with dynamic exploration parameter.
    LCB = -mu + beta * sigma, where beta adapts to the relative uncertainty.
    
    Parameters:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP std dev
        f_best: scalar - best observed value
        cost: array of shape (2000,) - cost of evaluating each candidate

    Returns:
        array of shape (2000,) - finite scores, higher = evaluate next
    """
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 330s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    n = len(mu_safe)
    
    # Compute the scale of the objective to normalize uncertainty
    # Use the range of mu as a proxy for the function's dynamic range
    mu_range = np.ptp(mu_safe) if n > 1 else 1.0
    if mu_range < 1e-9:
        mu_range = 1.0
    
    # Dynamic beta: scale with relative uncertainty
    # When sigma is large relative to the function's range, explore more
    # When sigma is small, exploit (lower beta)
    sigma_mean = np.mean(sigma_safe)
    rel_uncertainty = sigma_mean / mu_range
    
    # Beta ranges from ~1.0 (exploit) to ~3.0 (explore)
    # Use a smooth transition based on relative uncertainty
    # When rel_uncertainty is small (< 0.1), beta ~ 1.0
    # When rel_uncertainty is large (> 0.5), beta ~ 3.0
    beta = 1.0 + 2.0 * np.tanh(5.0 * rel_uncertainty)
    
    # LCB for minimization: we want to minimize, so we prefer low values
    # Score = -mu + beta * sigma (higher is better for selection)
    lcb_scores = -mu_safe + beta * sigma_safe
    
    # Cost-aware: divide by cost to prefer cheap evaluations
    # cost is in [1, 2], so this is a mild preference
    cost_adjusted_scores = lcb_scores / cost_safe
    
    # Ensure finite
    cost_adjusted_scores = np.where(np.isfinite(cost_adjusted_scores), cost_adjusted_scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return cost_adjusted_scores
    
    # Alternative: also consider a pure EI/cost version and compare
    if n >= 100 and time.time() < deadline - 10.0:
        z = (f_best - mu_safe) / sigma_safe
        ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        ei_cost = ei / cost_safe
        ei_cost = np.where(np.isfinite(ei_cost), ei_cost, 0.0)
        
        # Compare: mean mu of top-10 points
        top_k = min(10, n // 10)
        if top_k >= 1:
            idx_lcb = np.argsort(cost_adjusted_scores)[-top_k:]
            idx_ei = np.argsort(ei_cost)[-top_k:]
            
            mean_mu_lcb = np.mean(mu_safe[idx_lcb])
            mean_mu_ei = np.mean(mu_safe[idx_ei])
            
            # Prefer the one with lower mean mu in top-k
            if mean_mu_ei < mean_mu_lcb - 1e-6:
                cost_adjusted_scores = ei_cost
    
    # Final safety
    cost_adjusted_scores = np.where(np.isfinite(cost_adjusted_scores), cost_adjusted_scores, 0.0)
    
    return cost_adjusted_scores
# EVOLVE-BLOCK-END
