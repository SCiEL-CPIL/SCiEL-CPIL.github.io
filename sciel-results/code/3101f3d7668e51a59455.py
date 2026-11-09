"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 320.0  # 10s safety margin before 330s kill

def _elapsed_ok():
    return time.time() < _DEADLINE

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    Score = -(mu - beta * sigma) = -mu + beta * sigma
    Higher score means more promising to evaluate next.
    
    mu, sigma: shape (2000,) in objective units
    f_best: lowest observed value
    Returns: shape (2000,) finite scores, higher = evaluate next
    """
    if not _elapsed_ok():
        return np.random.rand(len(mu))
    
    # Constant beta for LCB; tuned for Michalewicz 2D with 30 evaluations
    # A moderate beta balances exploration and exploitation
    beta = 2.0
    
    # LCB score: we want to minimize, so higher score = lower (mu - beta*sigma)
    # score = -(mu - beta * sigma) = -mu + beta * sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    score = -mu + beta * sigma_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
