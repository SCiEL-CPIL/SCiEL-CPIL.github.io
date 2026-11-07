"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track elapsed time for adaptive beta decay
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    Args:
        mu: Mean of the Gaussian Process posterior, shape (2000,)
        sigma: Standard deviation of the GP posterior, shape (2000,)
        f_best: Best (lowest) observed function value so far.
        
    Returns:
        scores: Array of UCB scores, shape (2000,). Higher is better.
    """
    # Ensure inputs are numpy arrays and handle potential edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety margin: ensure sigma is not zero to avoid division issues or undefined behavior
    # Although UCB doesn't divide by sigma, extremely small sigma can cause numerical issues in some contexts
    # but here we just add it. We clip sigma to a small positive value to be safe.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Determine progress of the optimization
    # Total evaluations allowed: 30
    # Initial random points: 5
    # So there are 25 acquisition-driven steps.
    # The first call to acquisition happens after 5 evaluations.
    # We need to estimate the current step. 
    # Since we don't have access to the step counter directly in the function signature,
    # we can estimate it based on elapsed time, but that's unreliable if the evaluator is slow.
    # Alternatively, we can use a fixed schedule based on the assumption that the function is called sequentially.
    # However, a more robust approach for a "different strategy" without state is difficult.
    # Let's use a static but aggressive decay or a heuristic based on f_best? No, f_best is not a progress indicator.
    
    # Actually, we can maintain a simple global counter.
    global _call_count
    if not hasattr(acquisition, '_call_count'):
        acquisition._call_count = 0
        
    # Each call corresponds to one step in the loop.
    # Step 0: after 5 initial points.
    # Step 1: after 6 points.
    # ...
    # Step 24: after 29 points.
    # Step 25: after 30 points (last evaluation).
    # So we have 25 calls.
    
    current_step = acquisition._call_count
    acquisition._call_count += 1
    
    # Total steps = 25
    total_steps = 25
    progress = min(current_step / total_steps, 1.0)
    
    # Beta decay: Start with high exploration, end with low exploration.
    # Hartmann 3D is multi-modal. Early exploration is crucial.
    # Beta(t) = beta_max * (1 - progress) + beta_min * progress
    # Let's try beta_max = 3.0, beta_min = 0.5
    # This encourages looking at high uncertainty areas early on.
    beta_max = 3.0
    beta_min = 0.5
    beta = beta_max * (1.0 - progress) + beta_min * progress
    
    # UCB score: mu + beta * sigma
    # We want to minimize the function, so we want to maximize the score.
    # High mu (good value) and high sigma (uncertainty) should be preferred.
    scores = mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
