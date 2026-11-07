import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Hybrid EI + adaptive exploration bonus for minimization.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return -mu

    # EI for minimization:
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu)*Phi(z) + sigma*phi(z)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (f_best - mu) / sigma
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Adaptive exploration bonus
    f_norm = f_best / (f_best + 3.0)
    # lambda: ~1.5 early (f_best large), ~0.3 late (f_best small)
    lam = 0.3 + 1.2 * f_norm

    score = ei + lam * sigma

    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    return score
