import time
import numpy as np
from scipy.special import erf

_START_TIME = time.time()
_DEADLINE = 310.0
_CALL_COUNT = 0

def _norm_cdf(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def _norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)

def acquisition(mu, sigma, f_best, cost):
    global _CALL_COUNT, _DEADLINE, _START_TIME
    
    # Deadline guard
    if (time.time() - _START_TIME) > _DEADLINE:
        mu_safe = np.asarray(mu, dtype=np.float64)
        cost_safe = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost_safe, 1e-9)
        return -mu_safe / cost_safe

    _CALL_COUNT += 1
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), np.maximum(sigma, 1e-8), 1e-8)
    cost = np.where(np.isfinite(cost), np.maximum(cost, 1e-8), 1.0)
    
    # Estimate progress
    # Budget 30, start cost ~7.5, avg eval cost ~1.5
    # Remaining budget ~22.5, ~15 evals
    p = min(1.0, _CALL_COUNT / 15.0)
    
    # Dynamic Beta: High exploration early, low later
    beta_max = 2.5
    beta_min = 0.5
    beta = beta_max * (1.0 - p) + beta_min * p
    
    # UCB for minimization: -mu - beta * sigma
    ucb = -mu - beta * sigma
    
    # Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma
    z_clipped = np.clip(z, -50, 50)
    ei = improvement * _norm_cdf(z_clipped) + sigma * _norm_pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Scale EI to be comparable to UCB
    # UCB is in objective units. EI is in objective units.
    # However, EI can be small. Let's normalize EI by the range of mu or a fixed scale.
    # Using std of mu as a scale factor for EI seems reasonable to balance magnitudes.
    mu_std = np.std(mu)
    if mu_std < 1e-6:
        mu_std = 1.0
    ei_scaled = ei / mu_std
    
    # Combine UCB and EI
    # UCB provides exploration/exploitation balance.
    # EI provides smooth gradient towards improvement.
    # Weighting: UCB is more aggressive for finding new minima.
    w_ucb = 0.6
    w_ei = 0.4
    raw_score = w_ucb * ucb + w_ei * ei_scaled
    
    # Cost-aware: Divide by cost
    cost_safe = np.maximum(cost, 1e-8)
    scores = raw_score / cost_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Tiny noise for tie-breaking (deterministic to avoid randomness issues if any)
    idx = np.arange(len(scores))
    scores += 1e-10 * np.sin(idx * 0.1)
    
    return scores
