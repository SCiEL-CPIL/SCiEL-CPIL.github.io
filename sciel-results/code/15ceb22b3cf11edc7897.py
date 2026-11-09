# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 300.0  # Leave margin before 330s kill


def _expected_improvement(mu, sigma, f_best):
    """Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-12)
    improvement = f_best - mu
    z = improvement / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = improvement * cdf + sigma_safe * pdf
    return ei, z, cdf


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    
    Uses a hybrid approach:
    - Early in optimization (f_best is high): emphasize exploration via UCB
    - Later in optimization (f_best is low): emphasize exploitation via EI
    
    The Beale function has a sharp minimum at (3, 0.5) with value 0.
    The function is relatively flat in many regions, so exploration is crucial
    to find the narrow basin of attraction.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Emergency fallback: pure exploitation
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute EI components
    ei, z, cdf = _expected_improvement(mu, sigma_safe, f_best)
    
    # UCB score for minimization: -mu + beta * sigma
    # Adaptive beta: higher when f_best is large (early), lower when f_best is small (late)
    # The Beale function's maximum is around 355, minimum is 0
    # Use a smooth decay based on how close we are to the optimum
    # When f_best > 10, we're likely far from optimum -> explore more
    # When f_best < 1, we're close -> exploit more
    # beta ranges from ~0.5 (exploit) to ~3.0 (explore)
    f_best_clamped = max(f_best, 1e-12)
    # Log-scale decay: beta = a + b / (1 + f_best)
    # At f_best=0: beta = a + b
    # At f_best=10: beta = a + b/11
    # At f_best=100: beta = a + b/101
    a, b = 0.3, 2.7
    beta = a + b / (1.0 + f_best_clamped)
    
    ucb = -mu + beta * sigma_safe
    
    # PI (Probability of Improvement) as an additional signal
    pi = cdf  # Probability that f(x) < f_best
    
    # Blend strategy based on optimization progress
    # Early: more UCB weight; Late: more EI weight
    # Use f_best as progress indicator
    # progress = 0 when f_best is large, 1 when f_best is near 0
    progress = 1.0 / (1.0 + f_best_clamped)
    
    # Weights: UCB dominates early, EI dominates late
    w_ucb = 0.6 * (1.0 - progress) + 0.2 * progress  # 0.6 early, 0.2 late
    w_ei = 0.4 * (1.0 - progress) + 0.6 * progress   # 0.4 early, 0.6 late
    w_pi = 0.0  # PI is already captured in EI
    
    # Normalize each component to have zero mean and unit std for fair blending
    def normalize(arr):
        std = np.std(arr)
        if std < 1e-12:
            return arr - np.mean(arr)
        return (arr - np.mean(arr)) / std
    
    ucb_norm = normalize(ucb)
    ei_norm = normalize(ei)
    
    score = w_ucb * ucb_norm + w_ei * ei_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
