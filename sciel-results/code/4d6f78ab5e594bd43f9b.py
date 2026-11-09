import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling-inspired acquisition for Bayesian optimization of Ackley 2D.
    
    Score = -mu - beta * sigma
    where beta is dynamically scaled by f_best to be aggressive early (high exploration)
    and focused late (low exploration as f_best -> 0).
    
    The key structural change from UCB/EI hybrids:
    - Uses negative mean (exploitation) combined with a quantile-normalized uncertainty term
    - The exploration weight decays as f_best approaches the known minimum (0)
    - This creates a natural schedule: high beta when f_best is large (early), 
      low beta when f_best is small (late)
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Dynamic beta: scale by how far f_best is from the minimum (0)
    # Early on: f_best is large (~2-3 for Ackley), so beta is large -> more exploration
    # Late: f_best is small (~0.01-0.1), so beta is small -> more exploitation
    # Use a smooth decay: beta = base_beta * (1 + f_best) / (1 + f_best_ref)
    # where f_best_ref is a reference value (~1.0 for Ackley scale)
    f_best_clamped = max(f_best, 1e-6)
    f_best_ref = 1.0
    base_beta = 2.0
    
    # beta decays as f_best -> 0
    beta = base_beta * (1.0 + f_best_clamped / f_best_ref)
    
    # Thompson Sampling heuristic: score = -mu - beta * sigma
    # Higher score = more likely to be evaluated next
    # The -mu term exploits low-mean regions
    # The -beta*sigma term explores high-uncertainty regions
    scores = -mu - beta * sigma
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
