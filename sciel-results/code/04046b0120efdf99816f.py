import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Dynamic hybrid: blend EI and LCB based on uncertainty level
    # LCB = mu - beta * sigma (for minimization, we want to minimize, so lower is better)
    # But since we want higher scores to mean "evaluate next", we use:
    # LCB_score = -mu + beta * sigma (higher means more exploration)
    
    # Use adaptive beta based on the relative uncertainty
    # When sigma is large relative to f_best, explore more
    # When sigma is small, exploit more (pure EI)
    
    # Normalize sigma to get a measure of uncertainty level
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # Adaptive beta: higher when overall uncertainty is high
    # beta ranges from ~0.1 (low uncertainty) to ~1.0 (high uncertainty)
    # Use a smooth transition based on sigma_mean relative to typical scale
    if sigma_max > 1e-10:
        uncertainty_level = sigma_mean / (sigma_max + 1e-10)
    else:
        uncertainty_level = 0.0
    
    # beta = 0.5 * (1 + uncertainty_level) gives range [0.5, 1.0]
    # But let's use a different approach: weight based on how "explored" we are
    # Early on (high sigma), use more LCB; later (low sigma), use more EI
    
    # Simple adaptive: blend = 0.3 + 0.7 * (1 - min(1, sigma_mean / 0.1))
    # This gives higher exploration weight when sigma_mean is high
    
    if sigma_mean > 0.1:
        blend = 0.3  # More EI when uncertainty is high? No, let me reconsider
    else:
        blend = 0.7  # More exploration when uncertainty is low? That doesn't make sense
    
    # Let me use a cleaner approach:
    # score = (1 - alpha) * EI + alpha * LCB_score
    # where LCB_score = -mu + beta * sigma
    # and alpha, beta are tuned
    
    # For Ackley 2D, which is multimodal but not extremely so, 
    # a moderate exploration bonus works well
    
    # Use fixed moderate values that balance exploitation and exploration
    beta = 0.5
    lcb_score = -mu + beta * sigma_safe
    
    # Blend: 70% EI + 30% LCB
    scores = 0.7 * ei + 0.3 * lcb_score
    
    # Deadline guard
    if time.time() - start_time > 1.0:
        # Fallback: Greedy (minimize mu)
        scores = -mu
        
    return scores
