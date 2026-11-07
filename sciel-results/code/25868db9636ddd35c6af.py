import numpy as np
from scipy.stats import norm
import time

# Track start time for deadline guard
_START_TIME = None

def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME

def _check_deadline():
    """Return True if we're getting close to the 330s deadline (with safety margin)."""
    if _START_TIME is None:
        return False
    elapsed = time.time() - _START_TIME
    return elapsed > 300  # 30s safety margin

def _ei_score(mu, sigma, f_best, cost):
    """Standard Expected Improvement for minimization, normalized by cost."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Normalize cost: cost is in [1, 2], so use cost as-is
    return ei / np.maximum(cost, 1e-9)

def _ucb_score(mu, sigma, f_best, cost):
    """Upper Confidence Bound (lower confidence bound for minimization) normalized by cost."""
    # LCB = mu - beta * sigma; we want to maximize -LCB = -mu + beta * sigma
    beta = 2.0
    ucb = -mu + beta * sigma
    return ucb / np.maximum(cost, 1e-9)

def _pi_score(mu, sigma, f_best, cost):
    """Probability of Improvement for minimization, normalized by cost."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    # Scale to make it comparable to EI
    pi_scaled = pi * np.maximum(f_best - mu, 0) + 1e-6
    return pi_scaled / np.maximum(cost, 1e-9)

def _hypervolume_score(mu, sigma, f_best, cost):
    """
    Multi-objective inspired acquisition: treat (predicted value, cost) as two objectives.
    
    We approximate hypervolume improvement by considering:
    1. How much the predicted value improves over f_best
    2. How much the cost is reduced over the average cost
    
    A point is good if it has low mu AND low cost.
    We compute a joint score that captures the area of the Pareto improvement.
    
    Key insight: The "reference point" is (f_best, mean_cost).
    For each candidate, the improvement in objective is max(f_best - mu, 0)
    and the improvement in cost is max(mean_cost - cost, 0).
    The hypervolume contribution is approximately the product of these two improvements.
    """
    mean_cost = np.mean(cost)
    
    # Improvement in objective (higher is better)
    obj_improve = np.maximum(f_best - mu, 0)
    
    # Improvement in cost (lower cost is better, so improvement is mean_cost - cost)
    cost_improve = np.maximum(mean_cost - cost, 0)
    
    # Hypervolume-like score: product of improvements
    # This rewards points that are both better than f_best AND cheaper than average
    hv = obj_improve * cost_improve
    
    # Also add a small term for just being good in one dimension
    # to avoid zero scores when one dimension has no improvement
    hv += 0.1 * obj_improve / np.maximum(cost, 1e-9)
    hv += 0.1 * cost_improve / np.maximum(cost, 1e-9)
    
    return hv

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a multi-objective (hypervolume-inspired) approach combined with
    standard EI and UCB terms, all normalized by cost.
    
    Parameters:
    mu: array of shape (N,)
    sigma: array of shape (N,)
    f_best: float, best observed value so far
    cost: array of shape (N,), cost of evaluating each candidate
    
    Returns:
    scores: array of shape (N,), higher is better
    """
    _ensure_start()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    N = len(mu)
    
    # Check deadline - if too close, return a simple fast score
    if _check_deadline():
        # Fast fallback: simple EI / cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / np.maximum(cost, 1e-9)
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Compute multiple acquisition scores
    score_ei = _ei_score(mu, sigma, f_best, cost)
    score_ucb = _ucb_score(mu, sigma, f_best, cost)
    score_pi = _pi_score(mu, sigma, f_best, cost)
    score_hv = _hypervolume_score(mu, sigma, f_best, cost)
    
    # Normalize each score to have zero mean and unit variance for fair blending
    def _normalize(s):
        s = np.asarray(s, dtype=np.float64)
        s = np.nan_to_num(s, nan=0.0, posinf=1e6, neginf=-1e6)
        s_mean = np.mean(s)
        s_std = np.std(s)
        if s_std < 1e-12:
            return s - s_mean
        return (s - s_mean) / s_std
    
    norm_ei = _normalize(score_ei)
    norm_ucb = _normalize(score_ucb)
    norm_pi = _normalize(score_pi)
    norm_hv = _normalize(score_hv)
    
    # Adaptive weights: explore more early, exploit more later
    # Use the range of mu as a proxy for uncertainty
    mu_range = np.ptp(mu)
    mu_mean = np.mean(mu)
    
    # If mu is very spread out, we're likely early -> explore more
    # If mu is concentrated, we're likely later -> exploit more
    if mu_range > abs(mu_mean) * 2 + 10:
        # Early phase
        w_ei = 0.25
        w_ucb = 0.35
        w_pi = 0.15
        w_hv = 0.25
    elif mu_range > abs(mu_mean) * 0.5 + 5:
        # Middle phase
        w_ei = 0.35
        w_ucb = 0.25
        w_pi = 0.15
        w_hv = 0.25
    else:
        # Late phase
        w_ei = 0.50
        w_ucb = 0.15
        w_pi = 0.10
        w_hv = 0.25
    
    # Weighted combination
    scores = w_ei * norm_ei + w_ucb * norm_ucb + w_pi * norm_pi + w_hv * norm_hv
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
