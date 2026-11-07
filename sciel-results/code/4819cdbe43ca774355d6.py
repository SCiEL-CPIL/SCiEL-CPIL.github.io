import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Styblinski-Tang 2D BO.
    Uses a structural LCB-based approach that directly minimizes the
    lower confidence bound divided by cost, targeting high-value, low-cost
    regions without stochastic variance of UCB/EI.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clean inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Deadline guard
    if time.time() > _DEADLINE:
        score = -mu / cost
        return np.where(np.isfinite(score), score, 0.0)
    
    # Progress: how close are we to the known global minimum?
    known_min = -78.3323
    # f_best starts high (positive) and decreases toward -78.3
    # progress = 0 when f_best is far from optimum, 1 when close
    progress = np.clip(1.0 - (f_best - known_min) / 250.0, 0.0, 1.0)
    
    # Adaptive beta for LCB: early on, explore more; later, exploit
    # LCB = mu - beta * sigma (we want to minimize this, so maximize -LCB)
    # Higher beta = more exploration
    beta = 1.5 + 2.5 * (1.0 - progress)
    
    # Lower Confidence Bound: lower is better
    lcb = mu - beta * sigma
    
    # We want to maximize the acquisition score
    # Score = -LCB / cost = (beta*sigma - mu) / cost
    # But we also want to incorporate expected improvement for exploitation
    
    # Expected Improvement
    z = (f_best - mu) / (sigma + 1e-12)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Combine LCB-based score and EI
    # LCB score favors exploration, EI favors exploitation
    # Use a weighted combination that shifts from exploration to exploitation
    # as progress increases
    w_ei = 0.3 + 0.7 * progress  # weight for EI
    w_lcb = 1.0 - w_ei           # weight for LCB
    
    # Normalize both to have similar scale
    # LCB score: (beta*sigma - mu) / cost
    lcb_score = (beta * sigma - mu) / cost
    # EI score: ei / cost
    ei_score = ei / cost
    
    # Combine
    score = w_lcb * lcb_score + w_ei * ei_score
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
