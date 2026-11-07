import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    # Handle numerical stability: avoid division by zero
    # If sigma is very small, the distribution is nearly deterministic.
    # We can treat it as a hard improvement if mu < f_best, else 0.
    # However, standard EI formula works if we handle the limit.
    # Let's add a small epsilon to sigma to prevent division by zero.
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    Z = (f_best - mu) / sigma_safe
    
    # Phi(Z) is the CDF of the standard normal distribution
    # phi(Z) is the PDF of the standard normal distribution
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    
    # Calculate EI
    # If sigma is effectively 0, Z is huge or -huge.
    # If mu < f_best, Z is large positive, Phi(Z) ~ 1, phi(Z) ~ 0. EI ~ f_best - mu.
    # If mu > f_best, Z is large negative, Phi(Z) ~ 0, phi(Z) ~ 0. EI ~ 0.
    # This behavior is correct.
    
    scores = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Ensure scores are finite
    # In rare cases, extreme values might cause overflow, but standard normal functions are stable.
    # Clip scores to a reasonable range to prevent any issues, though unlikely needed.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
