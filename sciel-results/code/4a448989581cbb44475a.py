"""Cost-aware acquisition for Ackley 2D BO with deadline guard and PI-based scoring."""
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
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    pi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * pdf(z)
    # pdf(z) = (1/sqrt(2*pi)) * exp(-0.5 * z^2)
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    ei = (f_best - mu) * pi + sigma * pdf_z
    
    # Ensure non-negative
    ei = np.maximum(ei, 0.0)
    
    # Combine EI and PI? 
    # Let's try pure EI/cost first, but using the faster erf calculation.
    # If EI is too greedy, PI might be better.
    # Given the "simple regret" metric, EI is standard.
    # I will stick to EI/cost but use the faster math.
    
    scores = ei / cost
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores
# EVOLVE-BLOCK-END
