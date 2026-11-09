# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    t0 = time.time()

    # Clamp sigma to a numerically safe floor
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)

    # Standardised distance of each candidate from the current best
    z = (f_best - mu) / sigma_safe

    # Probability of Improvement for minimization:
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)

    # Add a small exploitation bias to break ties among points with similar PI
    # This gently prefers lower mean values without overwhelming the PI signal
    exploit_bias = 0.1 * (f_best - mu) / (np.abs(f_best - mu).max() + 1e-12)
    
    # Blend PI with a mild exploitation term
    score = pi + exploit_bias

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard: if we are close to the 330 s wall-clock limit, return immediately
    if time.time() - t0 > 310.0:
        return score

    return score
# EVOLVE-BLOCK-END
