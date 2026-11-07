import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290.0 # 40s safety margin from 330s limit
    
    # Expected Improvement (EI) for minimization
    # E[I] = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Z = (f_best - mu) / sigma
    
    scores = np.zeros_like(mu)
    
    # Mask for points with significant uncertainty
    # If sigma is very small, the distribution is effectively a point mass
    mask = sigma > 1e-8
    
    if np.any(mask):
        mu_m = mu[mask]
        sigma_m = sigma[mask]
        
        # Compute Z
        z = (f_best - mu_m) / sigma_m
        
        # Compute EI
        # Use scipy.stats.norm for numerical stability
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        
        ei = (f_best - mu_m) * Phi_z + sigma_m * phi_z
        
        scores[mask] = ei
        
    # For points with negligible sigma, if mu < f_best, EI is 0 (no improvement expected from noise)
    # But if we are minimizing, and mu is lower than f_best, we've already found a better point?
    # Wait, f_best is the *lowest observed value*.
    # If mu < f_best, it implies the GP predicts a value lower than the best seen so far.
    # In EI, if mu < f_best, Z > 0.
    # If sigma is 0, the value is deterministic. If mu < f_best, the improvement is f_best - mu.
    # However, usually EI is defined for the *potential* improvement.
    # If sigma is 0, the uncertainty is gone.
    # Standard EI implementation often sets EI=0 where sigma=0 to avoid division by zero,
    # but technically if mu < f_best, we should prefer it.
    # However, since f_best is the best *observed*, and the GP is trained on observations,
    # mu at observed points is close to f(x).
    # Let's stick to the standard mask approach.
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
        
    return scores
