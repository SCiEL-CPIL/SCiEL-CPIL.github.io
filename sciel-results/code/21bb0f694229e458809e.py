"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Explicit wall-clock deadline guard (evaluator kills at 330 s; keep margin).
_DEADLINE = time.time() + 325.0


def acquisition(mu, sigma, f_best):
    # Time guard: if we are too close to the deadline, fall back to pure exploitation
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=float)
        return -mu
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Robustify sigma: clip tiny values and drop non-finite entries
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.clip(sigma, 0.0, None)
    
    # Base score: Expected improvement magnitude (f_best - mu)
    # Exploration bonus: beta * sigma
    # Beta = 1.5 encourages moderate exploration for Ackley's many local minima
    beta = 1.5
    
    score = (f_best - mu) + beta * sigma
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
