"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a fixed, aggressive Lower Confidence Bound (LCB).
    
    Strategy:
    1. Instead of selecting among multiple acquisition functions (EI, UCB, etc.), 
       we use a single, deterministic LCB score.
    2. The LCB is defined as: score = -mu + kappa * sigma.
    3. We choose a relatively high kappa (e.g., 3.0) to encourage exploration,
       which is often beneficial in early stages or when the posterior is uncertain,
       and avoids the pitfalls of meta-selection heuristics that might pick
       overly exploitative or inconsistent functions.
    4. This is a structural change from the "portfolio selection" approach to a
       "single robust exploration" approach.
    5. We include a time guard to ensure we return before the deadline.
    """
    start_time = time.time()
    deadline = start_time + 280  # 280s out of 300s to be safe
    
    n = len(mu)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Fixed aggressive exploration parameter
    # A higher kappa favors points with high uncertainty (sigma).
    # 3.0 is a strong exploration bias, often effective in 1D problems 
    # with a limited budget (25 evaluations) to cover the domain.
    kappa = 3.0
    
    # Compute LCB scores: we want to maximize -mu + kappa * sigma
    # Since higher score means "evaluate next", and we minimize the objective,
    # we want low mu (good prediction) and high sigma (uncertainty).
    scores = -mu + kappa * sigma
    
    # Check time
    if time.time() > deadline:
        # If we somehow ran out of time (unlikely for this simple op), return zeros
        # to ensure validity.
        return np.zeros(n)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END
