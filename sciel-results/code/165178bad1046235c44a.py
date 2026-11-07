import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin below 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive Cost-Aware LCB.
    Returns scores of shape (2000,), where higher scores indicate better candidates.
    """
    # Ensure safety in denominators
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Adaptive Beta for LCB
    # Cosine8 range is approx [-0.8, 1.0]. 
    # Initial f_best is likely near 0.5-1.0.
    # We want high beta when f_best is poor (high), low beta when f_best is good (low).
    
    # Estimate progress based on f_best
    # If f_best is 1.0 (worst), progress is 0.
    # If f_best is -0.8 (best), progress is 1.
    # Assuming initial uniform random points yield f_best ~ 0.5 to 1.0.
    # Let's define a reference worst case of 1.0 and best case of -0.8.
    
    # To be robust to different initializations, we can use a relative decay.
    # However, f_best is the only state variable passed.
    
    # Let's use a fixed but generous beta that decays slightly with improvement.
    # Or simply a well-tuned constant beta, as "adaptive" might be overkill if f_best is noisy.
    # But the hint says "adaptive Cost-aware LCB".
    
    # Let's try: beta = 1.0 + 2.0 * exp(-(1.0 - f_best) / 0.5)
    # If f_best = 1.0, beta = 3.0.
    # If f_best = -0.5, improvement = 1.5, exp(-3) ~ 0.05, beta ~ 1.1.
    # If f_best = -0.8, improvement = 1.8, exp(-3.6) ~ 0.02, beta ~ 1.04.
    
    # This encourages strong exploration at the start and exploitation later.
    
    # Calculate improvement signal
    # Clamp improvement to avoid negative values if f_best < 1.0 (which is always true for Cosine8 max 1.0)
    # Actually, if f_best is already very good, beta should be low.
    
    # Let's use a simpler adaptive scheme based on the current best value relative to the expected minimum.
    # min_possible = -0.8
    # max_possible = 1.0
    # progress = (max_possible - f_best) / (max_possible - min_possible)
    # progress is in [0, 1] roughly.
    
    max_val = 1.0
    min_val = -0.8
    
    # If f_best is very high (bad), progress is low.
    # If f_best is very low (good), progress is high.
    progress = (max_val - f_best) / (max_val - min_val)
    
    # Clamp progress to [0, 1] to be safe
    progress = np.clip(progress, 0.0, 1.0)
    
    # Beta starts at 3.0 and decays to 0.5
    beta_max = 3.0
    beta_min = 0.5
    
    # Exponential decay of exploration
    # When progress=0 (f_best=1.0), beta=beta_max
    # When progress=1 (f_best=-0.8), beta=beta_min
    beta = beta_min + (beta_max - beta_min) * np.exp(-3.0 * progress)
    
    # LCB for minimization: - (mu - beta * sigma)
    # We want to MAXIMIZE the score.
    # LCB value is lower for better candidates.
    # So Score = -LCB = mu - beta * sigma ? No.
    # LCB(x) = mu(x) - beta * sigma(x).
    # We want to minimize LCB.
    # So Acquisition Score = -LCB(x) = -mu(x) + beta * sigma(x).
    
    lcb_scores = -mu + beta * sigma_safe
    
    # Cost-Aware: Divide by cost
    # Higher cost reduces the score.
    scores = lcb_scores / cost_safe
    
    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
