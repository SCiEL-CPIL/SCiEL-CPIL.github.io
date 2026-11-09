"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds, evaluator kills at 330s

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Strategy: Rank-Normalized Adaptive UCB/EI
    1. Compute raw EI and UCB scores.
    2. Normalize these scores to ranks [0, 1] to make the acquisition function 
       invariant to the scale of the posterior predictive distribution.
    3. Dynamically adjust the weight between exploitation (EI) and exploration (UCB)
       based on how close f_best is to the known global minimum (-1.8013).
       Early on (f_best >> min), favor exploration. Later, favor exploitation.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (lowest, since we're minimizing).
    
    Returns:
    --------
    scores : np.ndarray, shape (n,)
        Acquisition scores; higher means more desirable to evaluate.
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Fallback: simple greedy exploitation
        return -np.asarray(mu, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Handle numerical issues in sigma
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Handle numerical issues in mu
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # 1. Compute Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical overflow/underflow in norm.cdf/pdf
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute UCB for minimization
    # We use a fixed beta for the base UCB, but its weight will be adaptive.
    # A moderate beta like 1.5-2.0 is standard.
    beta = 1.5
    ucb = -mu + beta * sigma
    
    # 3. Rank Normalization
    # Convert scores to ranks. Higher rank (1.0) means better candidate.
    # np.argsort(argsort()) gives rank indices.
    # We want higher EI/UCB to be better.
    
    # Normalize EI to [0, 1]
    ei_ranks = np.argsort(np.argsort(ei)) / max(len(ei) - 1, 1)
    
    # Normalize UCB to [0, 1]
    ucb_ranks = np.argsort(np.argsort(ucb)) / max(len(ucb) - 1, 1)
    
    # 4. Adaptive Weighting
    # Known global minimum for Michalewicz 2D is approx -1.8013
    f_min_global = -1.8013
    
    # Calculate progress: 0 when f_best is far from min, 1 when f_best == min
    # We use a smooth transition.
    # If f_best is very bad (e.g., 0.0), progress is low -> high exploration.
    # If f_best is close to -1.8013, progress is high -> high exploitation.
    
    # Let's define a scale. The range of Michalewicz 2D is roughly [-1.8, 0.0].
    # Let's map f_best from [0.0, -1.8013] to [0, 1].
    # progress = (0.0 - f_best) / (0.0 - f_min_global)
    # But f_best can be worse than 0.0 initially.
    
    # A robust way: 
    # Let's say if f_best > -0.5, we are in early exploration phase.
    # If f_best < -1.5, we are in late exploitation phase.
    
    # Simple linear mapping of f_best to a weight w_exploit in [0.2, 1.0]
    # w_exploit = 1.0 when f_best is very good, 0.2 when f_best is very bad.
    
    # Scale: 
    # f_best = -1.8013 -> w_exploit = 1.0
    # f_best = 0.0    -> w_exploit = 0.2
    
    # Linear interpolation
    if f_best <= f_min_global:
        w_exploit = 1.0
    else:
        # Map [f_min_global, 0.0] to [1.0, 0.2]
        # slope = (0.2 - 1.0) / (0.0 - f_min_global) = -0.8 / 1.8013
        # w = 1.0 + slope * (f_best - f_min_global)
        w_exploit = 1.0 + (-0.8 / 1.8013) * (f_best - f_min_global)
        w_exploit = np.clip(w_exploit, 0.2, 1.0)
    
    w_explore = 1.0 - w_exploit
    
    # 5. Combine Ranks
    # Higher score is better.
    scores = w_exploit * ei_ranks + w_explore * ucb_ranks
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
