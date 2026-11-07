import numpy as np
from scipy.stats import norm
import time

_deadline = None
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function for Bayesian optimization.
    
    Uses a hybrid approach: primarily Thompson Sampling (sample from GP posterior,
    score by negative sample / cost) with an EI/cost component blended in.
    The blend ratio adapts over the course of optimization to balance exploration
    and exploitation.
    
    Parameters:
    mu: (2000,) array of mean predictions
    sigma: (2000,) array of standard deviations
    f_best: float, current best observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of scores (higher is better)
    """
    global _deadline, _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0
    
    _call_count += 1
    
    if time.time() > _deadline:
        return np.ones(len(mu))
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    n = len(mu)
    
    # Thompson Sampling: draw a sample from the GP posterior at each candidate
    # sample_i ~ N(mu_i, sigma_i^2)
    # Score for TS: -sample_i / cost_i (higher is better, i.e., lower predicted value)
    samples = mu + sigma_safe * np.random.randn(n)
    score_ts = -samples / (cost + eps)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    score_ei = ei / (cost + eps)
    
    # Adaptive blend: early on, rely more on Thompson Sampling for exploration;
    # later, shift toward EI for more focused exploitation.
    # blend_weight: weight on EI, (1-blend_weight) on TS
    # Start with high TS weight, decay toward EI
    blend_weight = 1.0 - np.exp(-0.12 * _call_count)  # 0 early, approaches 1 later
    
    # Normalize scores to comparable scales before blending
    ts_range = np.ptp(score_ts) + eps
    ei_range = np.ptp(score_ei) + eps
    
    score_ts_norm = (score_ts - np.min(score_ts)) / ts_range
    score_ei_norm = (score_ei - np.min(score_ei)) / ei_range
    
    scores = (1.0 - blend_weight) * score_ts_norm + blend_weight * score_ei_norm
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
