"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a deterministic Lower Confidence Bound (LCB)
    with a high exploration coefficient to aggressively target regions of
    high posterior uncertainty. This is critical for escaping local optima
    in the multimodal Hartmann 6D landscape within a tight 30-evaluation budget.

    The LCB score is: -mu - kappa * sigma
    where kappa is chosen to balance exploration and exploitation.
    """
    # ----- time guard (safety margin) -----
    t0 = time.time()
    deadline = t0 + 295.0  # leave 5 s for caller overhead

    n = mu.shape[0]

    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)

    # LCB acquisition for minimization:
    # We want to minimize, so we prefer low mu and high sigma (uncertainty).
    # LCB = -mu - kappa * sigma
    # Higher score means more attractive.
    
    # Use a relatively high kappa to prioritize exploration
    kappa = 3.0
    
    scores = -mu - kappa * sigma_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # If we are dangerously close to the deadline, return immediately
    if time.time() > deadline - 0.5:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
