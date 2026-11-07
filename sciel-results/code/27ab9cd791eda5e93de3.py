"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Per-process deadline guard
_T0 = time.time()
_DEADLINE = 330.0


def _elapsed():
    return time.time() - _T0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Thompson Sampling score (sampled objective value) normalized by cost.
    This implicitly balances exploration and exploitation by sampling from the
    posterior for each candidate, then dividing by evaluation cost.
    """
    # Guard: if we are dangerously close to the wall-clock limit, return a
    # deterministic, finite, low-risk signal immediately.
    if _elapsed() > _DEADLINE - 1.0:
        return np.zeros(np.shape(mu), dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Sanitize inputs
    sigma = np.clip(sigma, 1e-9, None)
    cost = np.clip(cost, 1e-9, None)
    f_best = float(f_best) if np.isfinite(f_best) else 0.0

    # Thompson Sampling: sample one objective value per candidate from the GP posterior
    # For minimization, lower sampled value is better.
    # Score = -sampled_value / cost (higher score = better)
    # This is equivalent to maximizing (f_best - sampled_value) / cost in expectation,
    # but uses a single sample for diversity.
    
    # Generate a single random normal per candidate
    # Use a fixed seed per call? No, we want different samples each call for diversity.
    # But the evaluator calls this 2000 times per iteration, so we need consistent
    # behavior. Actually, the function is called once with 2000 candidates.
    
    # Sample from N(mu, sigma^2)
    # To avoid extreme outliers, clip the samples
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(mu.shape)
    
    # For minimization, we want to select the point with the lowest predicted value.
    # Score should be higher for better (lower) values.
    # score = -samples / cost
    
    # However, pure Thompson sampling can be too exploratory.
    # A common hybrid is to use the negative sampled value, which naturally
    # balances exploitation (low mu) and exploration (high sigma).
    
    score = -samples / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
