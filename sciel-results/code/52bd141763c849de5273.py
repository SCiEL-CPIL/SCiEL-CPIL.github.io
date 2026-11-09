"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: return early if we're running too long
_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill


def _time_check():
    """Check if we're close to the deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a hybrid approach combining:
    - Probability of Improvement (POI): P(f < f_best) for robust exploitation
    - UCB exploration term for mapping the landscape
    
    The blend adapts based on f_best magnitude:
    - Early phase (f_best large): more exploration via UCB
    - Late phase (f_best small): more exploitation via POI
    """
    # Check deadline
    if _time_check():
        # Return a simple fallback: UCB with beta=0.5
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
        mu_safe = np.asarray(mu, dtype=np.float64)
        return 0.5 * sigma_safe - mu_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma = np.maximum(sigma, 1e-8)

    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 1.0
    f_best = max(abs(f_best), 1e-10)

    # Compute Probability of Improvement (POI): P(f < f_best)
    # This is the CDF of the posterior at f_best
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    poi = norm.cdf(z)
    poi = np.where(np.isfinite(poi), poi, 0.0)

    # Compute UCB score for exploration
    # Adaptive beta: higher when f_best is large (early), lower when f_best is small (late)
    # For Ackley 2D on [-5,5]^2, values range roughly [0, ~20]
    # beta decays as f_best approaches 0
    beta = 0.8 * np.exp(-f_best / 2.0) + 0.2
    ucb = beta * sigma - mu
    ucb = np.where(np.isfinite(ucb), ucb, mu)

    # Blend strategy:
    # - POI is in [0, 1], so scale it to be comparable with UCB
    # - UCB can be positive or negative
    # - We want to combine them meaningfully
    
    # Normalize POI to have similar scale as UCB
    # POI ranges [0, 1], UCB can be large in magnitude
    # Scale POI by a factor that depends on the typical UCB magnitude
    ucb_scale = np.std(ucb) + 1e-12
    poi_scaled = poi * ucb_scale

    # Adaptive blend:
    # exploration_weight goes from ~0.5 (early) to ~0.1 (late)
    exploration_weight = 0.5 * np.exp(-f_best / 1.0) + 0.1

    # Final score: POI (scaled) + exploration bonus
    scores = poi_scaled + exploration_weight * ucb

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
