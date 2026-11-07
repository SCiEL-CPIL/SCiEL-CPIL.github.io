import numpy as np
from scipy.stats import norm
import time

_acq_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _acq_start_time
    if _acq_start_time is None:
        _acq_start_time = time.time()
    
    # Deadline guard: if we've been running too long, return a safe fallback
    elapsed = time.time() - _acq_start_time
    if elapsed > 280:
        # Safe fallback: prioritize low cost, then low mu
        scores = -mu / (cost + 1e-9)
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e6, neginf=1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Adaptive beta: start with more exploration, shift to exploitation
    # Budget is 30 total, we track progress via cost structure
    mean_cost = np.mean(cost)
    # Use f_best as a proxy for progress - lower f_best means more progress
    # But we don't know absolute scale, so use a fixed schedule based on elapsed
    progress = min(elapsed / 280.0, 1.0)
    beta = 2.0 * (1.0 - 0.5 * progress) + 0.3  # ranges from ~2.3 to ~1.3
    
    # Cost-aware EI: standard EI scaled by 1/cost
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # numerical stability
    
    # Standard EI for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus weighted by uncertainty and cost
    # This helps in early stages when EI is small
    exploration_bonus = beta * sigma / (cost ** 0.5)
    
    # Combine: EI/cost + exploration bonus
    scores = ei / cost + exploration_bonus * 0.1
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
