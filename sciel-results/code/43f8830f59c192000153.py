# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time for deadline guard
_START_TIME = time.time()
DEADLINE_MARGIN = 30.0  # Safety margin before 330s limit

def _elapsed():
    return time.time() - _START_TIME

def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs (1 + mean(x_scaled))
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're running low on time, return simple EI/cost quickly
    if _elapsed() > DEADLINE_MARGIN:
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / (cost + 1e-12)
        return np.where(np.isfinite(score), score, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Adaptive parameters based on progress
    f_scale = abs(f_best)
    if f_scale > 10.0:
        # Early phase: strong exploitation
        xi = 0.0
        w_ei = 0.55
        w_ts = 0.35
        w_explore = 0.10
    elif f_scale > 1.0:
        # Middle phase: balanced
        xi = 0.01
        w_ei = 0.40
        w_ts = 0.40
        w_explore = 0.20
    else:
        # Late phase: exploration focused
        xi = 0.05
        w_ei = 0.25
        w_ts = 0.45
        w_explore = 0.30
    
    # 1. Expected Improvement per unit cost
    ei = _expected_improvement(mu, sigma_safe, f_best, xi=xi)
    ei_cost = ei / (cost + 1e-12)
    
    # 2. Thompson Sampling: random Gaussian perturbation to GP mean
    #    This introduces stochastic exploration that escapes local minima
    #    in the acquisition landscape
    rng = np.random.default_rng()
    ts_samples = mu + sigma_safe * rng.standard_normal(len(mu))
    # For minimization, we want low values, so we negate to make "higher is better"
    ts_score = -ts_samples / (cost + 1e-12)
    
    # Normalize Thompson score to be comparable with EI
    ts_min = np.min(ts_score)
    ts_max = np.max(ts_score)
    if ts_max > ts_min + 1e-12:
        ts_norm = (ts_score - ts_min) / (ts_max - ts_min)
    else:
        ts_norm = np.zeros_like(ts_score)
    
    # 3. Cost-efficiency of exploration: high sigma / low cost
    cost_exp = sigma_safe / (cost + 1e-12)
    cost_exp_max = np.max(cost_exp) if len(cost_exp) > 0 else 1.0
    cost_exp_norm = cost_exp / (cost_exp_max + 1e-12)
    
    # Combine with adaptive weights
    score = w_ei * ei_cost + w_ts * ts_norm + w_explore * cost_exp_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
