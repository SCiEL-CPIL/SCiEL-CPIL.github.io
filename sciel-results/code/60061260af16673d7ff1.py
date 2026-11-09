import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check if we're approaching the deadline (330s limit, use 320s as safety margin)
    elapsed = time.time() - _start_time
    if elapsed > 320:
        safe_cost = np.maximum(cost, 1e-10)
        return -mu / safe_cost
    
    # Small epsilon to prevent division by zero
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    safe_cost = np.maximum(cost, eps)
    
    # Calculate z-score
    z = (f_best - mu) / safe_sigma
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of Improvement (PI)
    pi_val = norm.cdf(z_clipped)
    
    # Expected Improvement (EI)
    pdf_val = norm.pdf(z_clipped)
    ei_val = (f_best - mu) * pi_val + safe_sigma * pdf_val
    
    # Normalize EI to be comparable to PI (both in [0, 1] roughly, though EI can be larger)
    # Using f_best as a scale factor makes the acquisition function scale-invariant w.r.t the objective value
    # If f_best is close to 0, EI is small. Normalizing by f_best prevents EI from dominating when f_best is large.
    # However, if f_best is very small, this might amplify noise. 
    # A safer normalization is by the current sigma or a fixed scale.
    # Let's use a hybrid: normalize EI by max(f_best, mean_sigma) to keep it bounded.
    scale_factor = max(abs(f_best), np.mean(safe_sigma) + eps)
    ei_norm = ei_val / scale_factor
    
    # Structural Change: Blend PI and Normalized EI
    # PI favors points likely to be better than current best.
    # EI favors points with large expected gain.
    # This blend (KG-like) balances exploration (via PI's sensitivity to mu) and exploitation (via EI's magnitude).
    # Weights: 0.5 for PI, 0.5 for EI_norm.
    # This reduces the tendency to chase high-variance outliers compared to pure EI.
    combined_score = 0.5 * pi_val + 0.5 * ei_norm
    
    # Cost-aware: Divide by cost
    # Using 1/cost is standard. 
    score = combined_score / safe_cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
