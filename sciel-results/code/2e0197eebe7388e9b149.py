"""Cost-aware acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_DEADLINE_START = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s kill


def _time_left():
    return _TIME_LIMIT - (time.time() - _DEADLINE_START)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid UCB/EI approach
    with dynamic exploration coefficient.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best (lowest) observed objective value
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Check deadline
    if _time_left() < 0.5:
        return np.zeros_like(mu)
    
    # Ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute total cost so far (approximate from the cost array mean)
    # cost[i] = 1 + mean(x_i) where x_i in [0,1]^3, so cost in [1, 2]
    # Total budget is 30. We estimate progress from f_best and cost structure.
    
    # Dynamic kappa: shift from exploitation to exploration as budget depletes
    # Estimate remaining budget fraction
    # Average cost per eval ~ 1.5, so ~20 evals total, start with 5, so ~15 remaining
    # Use cost array to estimate: if costs are around 1.5, total spent ~ n_evals * 1.5
    # We don't have n_evals directly, but we can use the spread of cost as a proxy
    # Better: use the fact that cost = 1 + mean(x), so mean cost ~ 1.5
    # Total budget 30, so roughly 20 evaluations possible
    # Start with 5 uniform points, so ~15 iterations
    
    # Use a heuristic: the variance of cost gives us little info about progress
    # Instead, use a fixed schedule based on the number of unique cost values
    # or simply use a moderate exploration that balances well
    
    # Let's use a dynamic kappa based on the uncertainty level
    # Higher uncertainty -> more exploration
    mean_sigma = np.mean(sigma_safe)
    
    # Normalize uncertainty: if mean_sigma is large relative to f_best scale, explore more
    # f_best is typically around -3 to -3.86 for hartmann3
    # Scale sigma relative to the range of mu
    mu_range = np.ptp(mu) if np.ptp(mu) > 1e-9 else 1.0
    
    # Dynamic kappa: between 1.0 and 3.0
    # When uncertainty is high, explore more
    kappa_base = 1.5
    # Adjust based on how far we are from the known best
    improvement_scale = max(f_best - np.min(mu), 1e-6)
    
    # UCB component: mu + kappa * sigma (we want to minimize, so we prefer low mu)
    # For minimization, UCB-like: score = -(mu - kappa * sigma) / cost
    # Or: score = (kappa * sigma - mu) / cost
    # Higher score = more attractive (lower mu, higher sigma)
    
    # EI component:
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei_cost = ei / np.maximum(cost, 1e-9)
    
    # UCB component for minimization:
    # We want to minimize, so attractive points have low mu and high sigma
    # Score = (kappa * sigma - mu) / cost
    # But mu is negative (hartmann values are negative), so -mu is positive
    # Let's use: score_ucb = (kappa * sigma - mu) / cost
    # This rewards high sigma and low mu (more negative mu = higher -mu)
    
    # Actually, for minimization:
    # UCB lower bound: mu - kappa * sigma (lower is better for minimization)
    # So we want to maximize: -(mu - kappa * sigma) = kappa * sigma - mu
    # Yes, score_ucb = (kappa * sigma - mu) / cost
    
    # Dynamic kappa: start with more exploration, shift to exploitation
    # Use the ratio of current best to typical mu values
    # If f_best is close to the minimum of mu, we're exploiting
    # If there's still room, explore
    
    # Simple heuristic: kappa decreases as f_best approaches min(mu)
    min_mu = np.min(mu)
    gap = f_best - min_mu
    if gap > 1e-6:
        # Normalize gap: if gap is large, explore more
        kappa = kappa_base + 1.0 * np.tanh(gap / (0.5 * mu_range + 1e-9))
    else:
        kappa = kappa_base * 0.5
    
    # Clip kappa to reasonable range
    kappa = np.clip(kappa, 0.5, 4.0)
    
    score_ucb = (kappa * sigma_safe - mu) / np.maximum(cost, 1e-9)
    
    # Blend EI and UCB
    # Weight: when EI is large (good exploitation opportunity), weight more toward EI
    # When EI is small, weight more toward UCB for exploration
    ei_max = np.max(ei_cost) if np.max(ei_cost) > 1e-9 else 1.0
    ei_normalized = ei_cost / (ei_max + 1e-9)
    
    ucb_max = np.max(score_ucb) if np.max(score_ucb) > 1e-9 else 1.0
    ucb_normalized = score_ucb / (ucb_max + 1e-9)
    
    # Blend: 0.6 * EI + 0.4 * UCB
    # This maintains exploitation while allowing exploration
    score = 0.6 * ei_normalized + 0.4 * ucb_normalized
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
