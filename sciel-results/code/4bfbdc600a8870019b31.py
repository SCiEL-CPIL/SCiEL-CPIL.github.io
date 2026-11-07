import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
_TIME_LIMIT = 315.0  # Safety margin from 330s

def acquisition(mu, sigma, f_best):
    # Check time
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        return np.full(len(mu), -1e10)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if mu.size == 0:
        return np.array([])

    # Ensure sigma is non-negative and finite
    sigma = np.where(np.isfinite(sigma), np.maximum(sigma, 0.0), 0.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)

    # Use Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For GP with mean mu and std sigma:
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF of standard normal, phi is PDF.
    
    # Handle edge cases
    if not np.isfinite(f_best):
        # If f_best is not finite, use a simple UCB-like score
        # For minimization, we want to minimize mu + kappa * sigma
        # Score = -(mu + kappa * sigma) = -mu - kappa * sigma
        # But we want higher scores to be better, so:
        # Actually, let's use -mu + kappa * sigma for exploration
        # Wait, for minimization, LCB = mu - kappa * sigma
        # We want to minimize LCB, so we maximize -LCB = -mu + kappa * sigma
        kappa = 2.0
        scores = -mu + kappa * sigma
        return np.where(np.isfinite(scores), scores, -1e10)

    # Compute z = (f_best - mu) / sigma
    # Handle sigma = 0 case
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e10)
    z = (f_best - mu) / sigma_safe

    # For sigma = 0, EI is simply max(f_best - mu, 0)
    # We'll compute EI for all points and handle sigma=0 separately
    
    # Use scipy for Phi and phi
    try:
        from scipy.stats import norm
        Phi = norm.cdf(z)
        phi = norm.pdf(z)
    except ImportError:
        # Fallback: use numpy approximation
        # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        # phi(z) = exp(-z^2/2) / sqrt(2*pi)
        Phi = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
        phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)

    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # But for sigma = 0, we want EI = max(f_best - mu, 0)
    ei = (f_best - mu) * Phi + sigma * phi
    
    # For points where sigma was very small (set to 1e10), 
    # the EI formula above may not be accurate.
    # Let's recompute for those points using the limit.
    small_sigma = sigma < 1e-10
    if np.any(small_sigma):
        # For sigma -> 0, EI -> max(f_best - mu, 0)
        ei[small_sigma] = np.maximum(f_best - mu[small_sigma], 0.0)

    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)

    # The score to return should be higher for better candidates.
    # EI is already higher when improvement is expected, so we return EI directly.
    # But we need to ensure it's finite.
    scores = np.where(np.isfinite(ei), ei, 0.0)

    return scores
