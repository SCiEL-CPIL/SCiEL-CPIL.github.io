import numpy as np

def acquisition(mu, sigma, f_best):
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Calculate adaptive beta for Lower Confidence Bound (LCB)
    # For minimization, we maximize: -mu - beta * sigma
    # We want higher beta (more exploration) when f_best is high (far from optimum)
    # and lower beta (more exploitation) when f_best is low (close to optimum).
    
    # f_best is the current minimum value found.
    # For sphere x^2, optimum is 0.
    # If f_best is large, we are far.
    
    # Heuristic: beta scales with the square root of the current best value.
    # Clip to a reasonable range [0.5, 5.0] to avoid extreme behavior.
    # 0.5 ensures some exploration even when very close.
    # 5.0 ensures strong exploration when far away.
    
    current_best = float(f_best)
    # If f_best is negative (shouldn't happen for sphere, but good for safety), treat as 0
    if current_best < 0:
        current_best = 0.0
        
    # Dynamic beta
    # Start with a base exploration factor
    beta = 2.0 * np.sqrt(current_best)
    
    # Clip beta
    beta = np.clip(beta, 0.5, 5.0)
    
    # Calculate LCB scores
    # We want to maximize the score, which corresponds to minimizing the value.
    # The "value" estimate is mu. The uncertainty is sigma.
    # We prefer low mu and high sigma (if we want to explore).
    # So score = -mu - beta * sigma
    scores = -mu - beta * sigma
    
    # Handle potential numerical issues (infinities/nans)
    # If sigma is 0, score is just -mu.
    # If mu or sigma are nan/inf, replace with large negative value to deprioritize.
    scores = np.nan_to_num(scores, nan=-1e10, posinf=-1e10, neginf=-1e10)
    
    return scores
