import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0

# Known global minimum for Styblinski-Tang 1D (as per task description)
F_MIN = -39.1662

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive UCB.
    
    Structural change: Replaces the fixed-weight EI+UCB hybrid with a pure
    Upper Confidence Bound (UCB) where the exploration parameter beta is
    adapted dynamically based on the current best value f_best relative to
    the known global minimum. This allows the algorithm to be highly
    exploratory when f_best is far from the optimum and highly exploitative
    when it is close.
    """
    # Guard against negative or zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the gap between current best and known global minimum
    gap = f_best - F_MIN
    
    # If we have already found a value better than or equal to the known min,
    # we should be purely exploitative (low beta).
    # If gap is large, we are far from the optimum, so we should be exploratory (high beta).
    
    # Define beta range
    BETA_MIN = 0.1
    BETA_MAX = 3.0
    
    # Scale the gap. 
    # Typical initial gaps might be 10-50.
    # We want beta to be BETA_MAX for gap > 20, and BETA_MIN for gap <= 0.
    GAP_SCALE = 20.0
    
    # Clip gap to [0, GAP_SCALE] for the linear ramp
    gap_clipped = np.clip(gap, 0.0, GAP_SCALE)
    
    # Linear interpolation for beta
    beta = BETA_MIN + (BETA_MAX - BETA_MIN) * (gap_clipped / GAP_SCALE)
    
    # Compute UCB Score for minimization:
    # We want to minimize mu - beta * sigma.
    # To maximize a score, we use: -mu + beta * sigma
    score = -mu + beta * sigma_safe
    
    # Normalize scores to [0, 1] range to maintain consistency with previous
    # program outputs and ensure finite, bounded values.
    # Note: Normalization does not change the argmax.
    score_min = np.min(score)
    score_max = np.max(score)
    score_range = score_max - score_min
    
    if score_range < 1e-12:
        score_norm = np.zeros_like(score)
    else:
        score_norm = (score - score_min) / score_range
        
    # Ensure finite output
    score_norm = np.where(np.isfinite(score_norm), score_norm, 0.0)
    
    return score_norm
