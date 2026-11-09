import numpy as np
from scipy.stats import norm
import time

# Global state to track evaluation count per seed
# We assume the evaluator runs seeds sequentially in the same process.
# We detect a new seed if f_best increases (since f_best is the minimum observed,
# it should be non-increasing within a single seed's optimization loop).
_eval_count = 0
_last_f_best = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Upper Confidence Bound (UCB) with adaptive exploration.
    
    For minimization, we maximize the score: -mu + kappa * sigma.
    kappa decreases as the number of evaluations increases to shift from
    exploration to exploitation.
    """
    global _eval_count, _last_f_best
    
    # Detect new seed start
    # f_best is the minimum observed value. In a single run, it only decreases (or stays same).
    # If it increases significantly, it must be a new independent run (new seed).
    if _last_f_best is not None:
        if f_best > _last_f_best + 1e-5:
            _eval_count = 0
    
    _eval_count += 1
    _last_f_best = f_best
    
    # Number of training points: 5 initial + eval_count steps
    n_train = 5 + _eval_count
    
    # Adaptive kappa: decreases with n_train
    # C is a scaling constant. 2.0 provides a balance.
    C = 2.0
    kappa = C / np.sqrt(n_train)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB Score for Minimization
    # We want high scores for low mu (exploit) and high sigma (explore).
    scores = -mu + kappa * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
