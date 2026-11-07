"""Bayesian optimization acquisition for Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

_obs_count = 5


def acquisition(mu, sigma, f_best):
    """
    LCB acquisition with time-decaying exploration for Beale 2D optimization.
    
    Uses a pure Lower Confidence Bound approach with a beta that starts high
    to explore the wide domain and decays to exploit promising regions.
    """
    global _obs_count
    t = _obs_count
    _obs_count += 1
    
    # Total evaluations: 5 initial + 25 optimization steps
    # t goes from 5 to 29 (0-indexed steps: 0 to 24)
    step = t - 5  # 0, 1, 2, ..., 24
    
    # Time-decaying beta: start high for exploration, decay to exploit
    # Beta starts at ~3.0 and decays to ~1.0
    beta = 3.0 * np.exp(-0.1 * step) + 1.0
    
    # LCB for minimization: maximize (beta * sigma - mu)
    lcb_score = beta * sigma - mu
    
    # Ensure finite values
    lcb_score = np.where(np.isfinite(lcb_score), lcb_score, 0.0)
    
    return lcb_score
# EVOLVE-BLOCK-END
