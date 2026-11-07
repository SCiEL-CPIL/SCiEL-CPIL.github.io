"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with
    cost-aware scaling and an exploration bonus.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from GP
    sigma : array-like, shape (2000,)
        Standard deviation predictions from GP
    f_best : float
        Best observed value so far (minimum since we minimize)
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values (higher = more promising)
    """
    # Deadline guard
    t_start = time.time()
    
    # Ensure arrays are 1-D numpy arrays of the right type
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.size
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Floor sigma to avoid division by zero / log of zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Potential improvement (we are minimizing, so improvement = f_best - mu)
    delta = f_best - mu
    
    # ---------- Expected Improvement (EI) ----------
    z = delta / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = delta * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)
    
    # ---------- Cost-aware scaling ----------
    # Use a cost penalty that's adaptive based on the mean cost
    mean_cost = float(np.mean(cost))
    
    # Normalize cost: costs range from 1 to 1 + mean(x_scaled) where x_scaled in [0,1]
    # So cost is in [1, 2]
    # We want to penalize higher costs, but not too aggressively
    # Use a power law: cost^(-alpha) where alpha adapts
    
    # Early on, be less cost-sensitive; later, be more cost-sensitive
    # norm_cost goes from 0 to 1 as mean_cost goes from 1 to 2
    norm_cost = np.clip((mean_cost - 1.0) / 1.0, 0.0, 1.0)
    
    # Alpha: start at 0.2 (mild penalty), increase to 0.6 (stronger penalty)
    alpha = 0.2 + 0.4 * norm_cost
    cost_weight = cost_safe ** (-alpha)
    
    # ---------- Exploration term ----------
    # UCB-style exploration: f_best - mu + kappa * sigma
    # Choose kappa based on uncertainty
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # High uncertainty relative to the scale of the problem suggests exploration
    # Use a fixed kappa that balances exploration and exploitation
    # kappa = 1.0 is a reasonable default for standardized problems
    kappa = 1.0
    
    ucb = f_best - mu + kappa * sigma_safe
    ucb = np.maximum(ucb, 0.0)
    
    # ---------- Combine EI and UCB ----------
    # Normalize both to [0, 1] range for combination
    def _normalise(arr):
        m = np.max(arr)
        if m > 1e-18:
            return arr / m
        return arr
    
    ei_n = _normalise(ei)
    ucb_n = _normalise(ucb)
    
    # Weight: start with more exploration, shift to exploitation as we learn
    # w_ei goes from 0.4 to 0.8, w_ucb goes from 0.6 to 0.2
    w_ei = 0.4 + 0.4 * norm_cost
    w_ucb = 1.0 - w_ei
    
    scores = w_ei * ei_n + w_ucb * ucb_n
    
    # Apply cost weighting
    scores = scores * cost_weight
    
    # ---------- Final normalisation for numerical stability ----------
    s_mean = np.mean(scores)
    s_std = np.std(scores)
    if s_std > 1e-12:
        scores = (scores - s_mean) / s_std
    else:
        scores = scores - s_mean
    
    # Guarantee finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
