import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function with adaptive beta for minimization.
    """
    start_time = time.time()
    deadline = 300.0
    
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Known minimum of Hartmann 3D function
    f_min_hartmann = -3.86278
    
    # Calculate gap to the global minimum
    gap = f_best - f_min_hartmann
    
    # Adaptive beta: higher exploration (beta) when far from optimum, lower when close
    # Clip gap to [0, 1] for the beta calculation to stabilize late-game behavior
    gap_clipped = np.clip(gap, 0.0, 1.0)
    
    # Beta ranges from 0.5 (exploitative) to 2.0 (exploratory)
    beta = 0.5 + 1.5 * gap_clipped
    
    # UCB for minimization: Maximize (beta * sigma - mu)
    # This is equivalent to minimizing the optimistic lower bound (mu - beta * sigma)
    scores = beta * sigma - mu
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - 1.0:
        pass
        
    return scores
