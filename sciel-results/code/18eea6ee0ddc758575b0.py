"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Thompson Sampling and Expected Improvement.
    
    Uses a hybrid approach:
    - Thompson Sampling: sample a single theta from N(mu, sigma) for each candidate
    - Blend with EI to get a robust acquisition
    - Normalize by cost to balance exploration vs exploitation
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Thompson Sampling: sample one theta per candidate from N(mu, sigma)
    # Use a fixed random state for reproducibility within a single call
    rng = np.random.default_rng(42)
    theta = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # For minimization, we want low values, so negative theta is good
    # TS score: lower theta is better, so we use -theta
    ts_score = -theta
    
    # Normalize both by cost
    ei_cost = ei / cost_safe
    ts_cost = ts_score / cost_safe
    
    # Blend: use a weighted combination
    # EI is more exploitation-focused, TS is more exploration-focused
    # In early iterations (high uncertainty), TS should dominate
    # In later iterations (low uncertainty), EI should dominate
    
    # Use the mean sigma as a proxy for uncertainty level
    mean_sigma = np.mean(sigma_safe)
    # If mean sigma is large, we're in early exploration phase
    # Scale the blend based on uncertainty
    # When mean_sigma is large, weight TS more; when small, weight EI more
    
    # Simple adaptive weight: w_ts = 1 / (1 + mean_sigma)
    # When mean_sigma ~ 1, w_ts ~ 0.5; when mean_sigma ~ 0.1, w_ts ~ 0.9
    # Actually, let's think differently:
    # In early stages, sigma is large, so we want more exploration (TS)
    # In late stages, sigma is small, so we want more exploitation (EI)
    
    # Use a sigmoid-like blend based on mean sigma
    # w_ts = 1 / (1 + exp(-mean_sigma))  # when mean_sigma large, w_ts ~ 1
    # Actually simpler: just use a fixed reasonable blend
    # Let's use 0.5 * EI + 0.5 * TS, both cost-normalized
    
    score = 0.5 * ei_cost + 0.5 * ts_cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
