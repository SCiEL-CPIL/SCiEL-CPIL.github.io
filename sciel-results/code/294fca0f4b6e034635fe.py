import numpy as np
from scipy.stats import norm
import time

# Global state for deadline management
_start_time = None
_deadline = 300.0
_safety_margin = 5.0

def _check_deadline():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    if time.time() > _start_time + _deadline - _safety_margin:
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Sanitize sigma to avoid division by zero
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Sanitize cost to avoid division by zero
    cost_safe = np.maximum(cost, eps)
    
    # 1. Compute Expected Improvement (EI) for minimization
    # We use the standard EI formulation.
    # Let improvement = f_best - mu.
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # PDF and CDF of standard normal
    pdf_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # EI = (f_best - mu) * P(X < f_best) + sigma * phi(z)
    # Note: For minimization, we want low values.
    # The standard EI formula for minimization is:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # However, if mu > f_best, (f_best - mu) is negative, and Phi(z) is small.
    # Let's stick to the robust definition:
    # EI = E[min(f_best - f(x), 0)] is not quite right for minimization score.
    # Standard EI minimization:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # 2. Exploration Term
    # The strategy directive suggested Thompson Sampling, but that was rejected as duplicate.
    # We will use a deterministic exploration term that rewards high uncertainty (sigma)
    # but penalizes high cost. This acts as a "confidence bound" style exploration.
    # UCB-like: score = -mu + beta * sigma / cost
    # But we also want exploitation of the EI.
    # Let's combine: Score = EI / cost + Exploration / cost
    # Exploration term: We use sigma^2 to make it quadratic in uncertainty, 
    # which is a common choice to balance the scales.
    # Let's try a weighted sum:
    # score = (EI + beta * sigma) / cost
    # However, EI already contains sigma. 
    # A distinct approach: Use Log-EI if possible, but EI is additive.
    # Let's use a "Cost-Adjusted UCB" combined with EI.
    # Actually, let's look at the "inspiration" 3 which mentioned "bounded, smooth exploration term".
    
    # Let's implement a score that is:
    # Score = EI / cost_safe
    # Plus an exploration boost for high sigma that is cheap.
    # boost = sigma / (1 + cost_safe)
    # But let's keep it simple and effective.
    # The previous program was simple EI / cost.
    # To be distinct, we will use:
    # Score = (EI + gamma * sigma^2) / cost_safe
    # where gamma is a small constant to encourage exploration in uncertain regions.
    
    gamma = 0.5
    exploration = gamma * (sigma_safe ** 2)
    
    raw_score = (ei + exploration) / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(raw_score), raw_score, -np.inf)
    
    # If deadline is near, we might want to return a simpler heuristic to save time?
    # The calculation above is vectorized and fast for 2000 points.
    # No special handling needed for deadline in the logic, just the global check if we were doing a loop.
    # Here we just compute and return.
    
    return scores
