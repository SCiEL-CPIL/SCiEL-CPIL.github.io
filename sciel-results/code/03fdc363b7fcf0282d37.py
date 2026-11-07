import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB for minimization
    # We want to maximize the score.
    # Low mu is good. High sigma is good (exploration).
    # Score = -mu + beta * sigma
    
    # Adaptive beta? 
    # Since we can't reliably track time, use a fixed beta.
    # Beta = 1.0 is a common start.
    # Let's try 1.5 to encourage exploration in multimodal landscape.
    
    beta = 1.5
    
    # Ensure sigma is not zero to avoid issues, though GP usually gives >0
    sigma_safe = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma_safe
    
    # The scores should be finite.
    # If mu is -inf or sigma is inf, handle it? 
    # The task says "Return 2000 finite scores".
    # If the GP produces NaNs, we should handle it.
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
