import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using a hybrid approach combining
    Expected Improvement (EI) with entropy-weighted exploration.
    
    Uses H-EI: multiplies EI by the differential entropy of the posterior,
    which amplifies improvement prospects in high-uncertainty regions while
    suppressing them in well-explored areas.
    """
    N = len(mu)
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute standard EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute differential entropy of the posterior: h = 0.5 * log(2*pi*e*sigma^2)
    # This is the entropy of a Gaussian with std sigma
    # h = 0.5 * (log(2*pi*e) + 2*log(sigma)) = 0.5*log(2*pi*e) + log(sigma)
    # Using log(sigma) as a proxy for entropy (monotonic transformation)
    log_sigma = np.log(sigma)
    
    # H-EI: multiply EI by entropy weight
    # The entropy weight should be positive and scale with uncertainty
    # Use a normalized version to avoid extreme values
    # Clip log_sigma to reasonable range
    log_sigma_clipped = np.clip(log_sigma, -10, 10)
    
    # Entropy weight: exp(log_sigma) = sigma, but we want a softer scaling
    # Use a power of sigma to control exploration
    # H-EI = EI * sigma^alpha where alpha controls exploration
    alpha = 0.5
    entropy_weight = np.power(sigma, alpha)
    
    # Normalize entropy weight to have mean 1 for stability
    mean_weight = np.mean(entropy_weight)
    if mean_weight > 1e-10:
        entropy_weight = entropy_weight / mean_weight
    else:
        entropy_weight = np.ones(N)
    
    # H-EI score
    he_i = ei * entropy_weight
    
    # Also compute UCB for comparison and potential blending
    # UCB = -mu + kappa * sigma (minimization, so higher is better)
    kappa = 1.0
    ucb = -mu + kappa * sigma
    
    # Normalize both to [0, 1] range for blending
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn > 1e-10:
            return (arr - mn) / (mx - mn)
        return np.zeros_like(arr)
    
    he_i_norm = normalize(he_i)
    ucb_norm = normalize(ucb)
    
    # Blend: 70% H-EI, 30% UCB
    blend_weight = 0.7
    scores = blend_weight * he_i_norm + (1 - blend_weight) * ucb_norm
    
    # Scale back to original EI scale for consistency
    # The absolute scale matters less than the relative ordering
    # But let's ensure scores are in a reasonable range
    scores = scores * (np.max(ei) - np.min(ei) + 1e-10)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
