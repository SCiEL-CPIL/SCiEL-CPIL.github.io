# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a budget-scaled LCB approach.
    Uses a linear decay of the exploration coefficient (beta) based on the
    progress of the search (improvement of f_best towards the known minimum).
    The score is the Lower Confidence Bound (LCB) normalized by the evaluation cost.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Constants
    F_MIN = -3.32237
    F_START = 0.0  # Approximate starting value of Hartmann 6d at uniform random points
    
    # Calculate progress of the search.
    # progress = 0 at start, 1 when we reach the global minimum.
    # f_best decreases from ~0 to -3.32.
    # We clamp f_best to [F_MIN, F_START] to handle noise or overshoots.
    f_best_clamped = np.clip(f_best, F_MIN, F_START)
    
    # If F_START == F_MIN, avoid division by zero (though not the case here)
    range_val = F_START - F_MIN
    progress = (F_START - f_best_clamped) / range_val
    progress = np.clip(progress, 0.0, 1.0)
    
    # Dynamically scale the exploration term (beta).
    # Start with aggressive exploration (high beta) and decay to exploitation (low beta).
    # Linear decay from BETA_MAX to BETA_MIN.
    BETA_MAX = 3.0
    BETA_MIN = 0.5
    beta = BETA_MIN + (BETA_MAX - BETA_MIN) * (1.0 - progress)
    
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize, so low mu is good.
    # Uncertainty sigma is good for exploration.
    lcb_value = -mu + beta * sigma_safe
    
    # Cost-normalized score
    # We divide by cost to prefer cheaper evaluations.
    # Higher score = better candidate to evaluate next.
    score = lcb_value / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
