"""Structural change: Thompson Sampling with cost-aware ranking."""
import time
import numpy as np

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s kill

# State for adaptive exploration
_call_count = 0
_last_f_best = None
_stagnation_counter = 0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Thompson Sampling with cost-adjusted ranking.
    
    Instead of deterministic EI, we sample from the GP posterior and rank
    candidates by a cost-adjusted score. This structural change from
    deterministic expected improvement to stochastic sampling can break
    out of local optima patterns.
    
    Args:
        mu: array shape (2000,) - GP mean predictions in objective units
        sigma: array shape (2000,) - GP std dev predictions in objective units
        f_best: float - lowest observed value
        cost: array shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array shape (2000,) - acquisition scores (higher = evaluate next)
    """
    global _call_count, _last_f_best, _stagnation_counter
    
    # Time guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        c = np.maximum(cost, 1e-9)
        return -mu / c
    
    _call_count += 1
    
    # Track stagnation: if f_best hasn't improved, increase exploitation
    if _last_f_best is not None:
        if f_best < _last_f_best - 1e-8:
            _stagnation_counter = 0
        else:
            _stagnation_counter += 1
    _last_f_best = f_best
    
    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)
    
    # Thompson Sampling: draw a sample from the posterior
    # Use a single random draw for efficiency
    rng = np.random.default_rng()
    sample = mu + s * rng.standard_normal(len(mu))
    
    # Adaptive kappa: reduce exploration when stagnant
    # kappa controls how much we trust the sample vs the mean
    # When stagnant, we want to exploit more (lower kappa)
    base_kappa = 0.3
    # Reduce exploration as stagnation increases, but keep some minimum
    kappa = max(base_kappa * (1.0 / (1.0 + 0.2 * _stagnation_counter)), 0.05)
    
    # Blend: use a convex combination of sample and mean
    # High kappa -> more exploration (closer to sample)
    # Low kappa -> more exploitation (closer to mean)
    blended = kappa * sample + (1.0 - kappa) * mu
    
    # Cost-aware scoring: we want low function values and low cost
    # Score = -blended / cost (higher is better)
    # This naturally favors cheap points with low predicted values
    score = -blended / c
    
    # Add a small uncertainty bonus to avoid completely ignoring exploration
    # Scale by 1/cost so cheap uncertain points get a boost
    uncertainty_bonus = 0.05 * s / c
    
    score = score + uncertainty_bonus
    
    return score
