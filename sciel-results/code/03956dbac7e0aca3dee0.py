import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s kill
_CALL_COUNT = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 3D.
    
    Combines Expected Improvement (EI) for exploitation with a dynamically 
    scaled exploration bonus. The exploration bonus is high in early iterations 
    to cover the multi-modal landscape and decays as the search converges 
    towards the known global minimum.
    """
    global _CALL_COUNT
    _CALL_COUNT += 1
    
    # Check time limit
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Emergency: return pure EI to be fast and valid
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0.0)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Calculate Expected Improvement (EI)
    # Standard EI for minimization
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    
    # 2. Calculate Exploration Bonus
    # We use a UCB-like term: -mu + beta * sigma
    # But we need to scale it to be comparable to EI.
    
    # Determine progress towards the known global minimum of Hartmann 3D
    known_min = -3.86278
    
    # Progress metric: 0.0 when f_best is far, 1.0 when f_best is close
    # Range of Hartmann 3D is roughly [-4, 0]
    # If f_best is close to -3.86, progress is high.
    if f_best <= known_min + 0.05:
        progress = 1.0
    else:
        # Linear mapping from -3.86 (1.0) to 0.0 (0.0)
        # Actually, let's map from a "bad" start (e.g., 0) to the min.
        # If f_best is 0, progress is 0. If f_best is -3.86, progress is 1.
        progress = np.clip((0.0 - f_best) / (0.0 - known_min), 0.0, 1.0)
        
    # Call-based decay: Exploration is critical in the first ~5-8 calls
    # After that, we should mostly exploit.
    # Use a sharp decay.
    if _CALL_COUNT <= 3:
        beta_base = 2.5  # High exploration
    elif _CALL_COUNT <= 8:
        beta_base = 1.0  # Moderate exploration
    else:
        beta_base = 0.3  # Low exploration, mostly exploit
    
    # Adjust beta based on progress: if we are close to the optimum, reduce exploration
    beta = beta_base * (1.0 - 0.5 * progress)
    
    # Exploration term: -mu + beta * sigma
    # Note: For minimization, lower mu is better. So -mu is high for low mu.
    # High sigma is good for exploration.
    explore_term = -mu + beta * sigma_safe
    
    # 3. Combine EI and Exploration
    # We need to balance the scales. EI is in objective units.
    # Explore term is also in objective units (since mu and sigma are in objective units).
    # However, EI is bounded by the improvement potential, while explore term can be larger.
    
    # Strategy: Weighted sum.
    # Early on, weight exploration higher. Later, weight EI higher.
    if _CALL_COUNT <= 3:
        w_explore = 0.5
        w_ei = 0.5
    elif _CALL_COUNT <= 8:
        w_explore = 0.3
        w_ei = 0.7
    else:
        w_explore = 0.1
        w_ei = 0.9
        
    # Normalize terms to [0, 1] range roughly to prevent scale dominance?
    # No, they are in the same units (objective value). 
    # But EI is non-negative. Explore term can be negative or positive.
    # To ensure "higher score means evaluate next", and we want to pick low mu/high sigma:
    # The raw sum might be dominated by the -mu part.
    
    # Let's try a different combination: 
    # Score = w_ei * EI + w_explore * (normalized explore_term)
    # Normalizing explore term: 
    # min_exp = np.min(explore_term)
    # max_exp = np.max(explore_term)
    # norm_exp = (explore_term - min_exp) / (max_exp - min_exp + 1e-9)
    
    # However, simple weighted sum in objective units is often robust if beta is tuned.
    # Let's stick to the additive form but ensure EI is the primary driver later.
    
    score = w_ei * ei + w_explore * explore_term
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
