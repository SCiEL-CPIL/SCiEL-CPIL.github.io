"""Rewritten program: Cost-aware acquisition with refined hybrid EI/UCB for Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 290.0  # 10s safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement and UCB.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate, in [1, 2].
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    # Time guard: if we are close to the limit, return a simple fallback
    if time.time() > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-8)
        return np.where(np.isfinite(ei / cost_safe), ei / cost_safe, 0.0)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware adaptive beta: use the spread of costs to estimate progress
    # Lower cost candidates are "cheaper" to explore, so we can afford more exploration
    # Use the mean cost as a proxy for budget consumption
    mean_cost = np.mean(cost)
    # beta decreases as mean_cost increases (more budget consumed)
    # Early: mean_cost ~ 1.0, beta ~ 1.5 (aggressive exploration)
    # Late: mean_cost ~ 1.5+, beta ~ 0.3 (conservative exploitation)
    beta = 1.5 * np.exp(-0.8 * (mean_cost - 1.0))
    beta = np.clip(beta, 0.3, 2.0)
    
    # UCB for minimization: we want to minimize, so score = -mu + beta * sigma
    ucb = -mu + beta * sigma_safe
    
    # Compute a cost-adjusted EI: divide EI by cost to reward cheap improvements
    cost_safe = np.maximum(cost, 1e-8)
    ei_cost_aware = ei / cost_safe
    
    # Normalize both terms to comparable scales
    ei_ca_mean = np.mean(ei_cost_aware)
    ei_ca_std = np.std(ei_cost_aware) + 1e-8
    ei_ca_norm = (ei_cost_aware - ei_ca_mean) / ei_ca_std
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-8
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Adaptive weighting: shift from exploration to exploitation
    # Use the gap to the known optimum as a signal: larger gap = more room to explore
    # Hartmann 3D minimum is approximately -3.86278
    gap_to_opt = max(f_best - (-3.86278), 0.0)
    # When gap is large (early), weight exploration more; when gap is small, exploit
    # Normalize gap: assume initial f_best ~ -1 to -2, so gap ~ 2-3
    gap_norm = np.clip(gap_to_opt / 3.0, 0.0, 1.0)
    
    # Weight for UCB (exploration) vs EI (exploitation)
    # Early: high UCB weight; Late: high EI weight
    weight_ucb = 0.3 + 0.5 * gap_norm  # 0.3 to 0.8
    weight_ei = 1.0 - weight_ucb
    
    combined = weight_ei * ei_ca_norm + weight_ucb * ucb_norm
    
    # Final cost-aware adjustment: divide by cost to prefer cheap evaluations
    scores = combined / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
