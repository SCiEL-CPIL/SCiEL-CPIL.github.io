"""Cost-aware acquisition for Ackley 2D BO with deadline guard and hybrid EI/PI scoring."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    if _check_deadline():
        mu_safe = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        cost_safe = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)
        return np.nan_to_num(-mu_safe / cost_safe, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)

    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement: Phi(z)
    pi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * pdf(z)
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    ei = (f_best - mu) * pi + sigma * pdf_z
    
    # Ensure non-negative
    ei = np.maximum(ei, 0.0)
    
    # Hybrid score: primarily EI, with a small PI bonus for exploration
    # Weight: 0.9 * EI + 0.1 * PI * sigma (scaled PI to be comparable)
    # This encourages exploring points with high uncertainty even if EI is modest
    hybrid = 0.9 * ei + 0.1 * pi * sigma
    
    # Cost-aware: divide by cost, but use a slightly softer penalty
    # cost is in [1, 2], so 1/cost ranges from 0.5 to 1.0
    scores = hybrid / cost
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores
# EVOLVE-BLOCK-END
