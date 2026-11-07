"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Combine Expected Improvement (EI) with an adaptive UCB exploration term.
    # EI handles exploitation (finding lower values near current best).
    # UCB term handles exploration (visiting uncertain regions), with
    # a coefficient that decays as f_best improves.
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: clamp sigma to avoid division by zero and handle non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    sigma = np.maximum(sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    # Clip z to avoid extreme values causing numerical issues
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive UCB coefficient:
    # - Starts high when f_best is large (early exploration)
    # - Decreases as f_best approaches 0 (later exploitation)
    # - Base of 0.3 ensures some exploration even at the end
    max_f = 25.0  # max value of x^2 on [-5, 5]
    ratio = np.clip(f_best / max_f, 0.0, 1.0)
    beta = 0.3 + 1.5 * np.sqrt(ratio)
    
    # UCB score for minimization: -mu + beta * sigma
    ucb = -mu + beta * sigma
    
    # Combine: weighted sum of EI and UCB
    # EI is in objective units, UCB is also in objective units
    # Use a fixed weight of 0.5 for UCB to balance exploration/exploitation
    score = ei + 0.5 * ucb
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
