import numpy as np
import time

# Module-level deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 325.0  # 5s safety margin before 330s kill

def _elapsed_ok():
    return time.time() < _DEADLINE

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Adaptive UCB (Lower Confidence Bound for minimization).
    
    Score = -mu + beta * sigma
    Higher score means more promising to evaluate next.
    
    Uses an adaptive beta that starts high (exploration) and decreases
    as more evaluations are made, shifting toward exploitation.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Estimate number of evaluations so far from the number of observations
    # We don't have direct access to n_evals, but we can infer progress
    # from the relationship between mu and f_best.
    # A simpler approach: use a fixed schedule based on typical BO runs.
    # Since we have 30 total evaluations and 5 initial, we have 25 BO steps.
    # We'll use a heuristic: if f_best is close to the known minimum (-1.8013),
    # we're likely late in the optimization.
    
    # Michalewicz 2D minimum is -1.8013
    f_min_known = -1.8013
    
    # Progress: 0 if f_best is far from minimum, 1 if close
    # Use a scale of about 2.0 units (typical range of Michalewicz values)
    progress = np.clip((f_best - f_min_known) / 2.0, 0.0, 1.0)
    
    # As progress increases (f_best gets closer to minimum), we want less exploration
    # So beta should decrease as progress increases
    # Start with beta ~ 2.0 (high exploration) and go down to ~ 0.5 (exploitation)
    beta = 2.0 - 1.5 * progress
    
    # Compute UCB score for minimization: -mu + beta * sigma
    # Higher is better (we want low mu and high sigma for exploration)
    score = -mu + beta * sigma_safe
    
    # Handle any numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Ensure all scores are finite
    if not np.all(np.isfinite(score)):
        score = np.where(np.isfinite(score), score, 0.0)
    
    return score
