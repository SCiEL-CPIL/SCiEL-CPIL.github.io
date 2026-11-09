import numpy as np
import time

# Global state for adaptive acquisition
_STEP = 0
_TOTAL_STEPS = 25  # 30 total evals - 5 initial

# Wall-clock deadline guard
_T0 = time.time()
_WALL_LIMIT = 330.0
_SAFETY = 45.0
_DEADLINE = _WALL_LIMIT - _SAFETY

def acquisition(mu, sigma, f_best):
    """
    Adaptive UCB acquisition function for minimizing Hartmann 6d.
    
    Structural change: Uses Upper Confidence Bound (UCB) with an adaptive
    exploration weight (beta) that decays over the optimization run.
    This prioritizes exploration in the early stages to cover the multimodal
    landscape and shifts to exploitation in later stages to refine the best
    candidate.
    
    Args:
        mu: (2000,) array of posterior means.
        sigma: (2000,) array of posterior standard deviations.
        f_best: float, the lowest observed value.
        
    Returns:
        (2000,) array of scores. Higher score means evaluate next.
    """
    global _STEP
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to simple greedy if time is running out
        return -mu
    
    # Update step counter
    _STEP += 1
    t = min(_STEP / _TOTAL_STEPS, 1.0)
    
    # Adaptive beta: Start with high exploration, end with low
    # Hartmann 6d is multimodal, so initial exploration is crucial.
    beta_start = 5.0
    beta_end = 1.0
    beta = beta_start * (1.0 - t) + beta_end * t
    
    # UCB for minimization: -mu + beta * sigma
    # We want to minimize f, so we maximize the score.
    # Low mu is good. High sigma is good (uncertainty).
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
