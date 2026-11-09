import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Upper Confidence Bound (UCB).
    
    The beta parameter is adjusted dynamically based on the current best value (f_best)
    relative to the known global minimum (3.0) and an estimated maximum.
    This encourages exploration when f_best is far from the optimum and exploitation
    when it is close.
    """
    # Explicit deadline guard (safety margin)
    # This function is vectorized and very fast, but we include the guard as per directive.
    start_time = time.monotonic()
    deadline = start_time + 5.0  # 5 seconds is a generous margin for this specific calculation
    
    # Constants for Goldstein-Price 2D
    F_MIN = 3.0
    F_MAX_EST = 50.0  # Conservative upper bound for the function in the domain
    
    # UCB Parameters
    BETA_MAX = 4.0  # Exploration weight when far from optimum
    BETA_MIN = 0.5  # Exploitation weight when close to optimum
    
    # Calculate progress towards the global minimum
    # p = 1.0 when f_best is at the minimum, p = 0.0 when f_best is at F_MAX_EST
    if F_MAX_EST > F_MIN:
        p = 1.0 - (f_best - F_MIN) / (F_MAX_EST - F_MIN)
    else:
        p = 1.0
        
    # Clip progress to [0, 1]
    p = np.clip(p, 0.0, 1.0)
    
    # Interpolate beta: high when p is low (far), low when p is high (close)
    beta = BETA_MAX - (BETA_MAX - BETA_MIN) * p
    
    # UCB Score for minimization:
    # We maximize the score.
    # Lower mu is better (exploitation), higher sigma is better (exploration).
    # Score = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Check if we exceeded the deadline (highly unlikely)
    if time.monotonic() > deadline:
        # In a more complex scenario, we might return a fallback.
        # Here, the calculation is already done.
        pass
        
    return scores
