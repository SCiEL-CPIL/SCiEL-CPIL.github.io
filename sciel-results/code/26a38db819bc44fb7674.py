import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a quantile-based normalization approach with dynamic weighting
    between exploitation and exploration based on the gap between f_best
    and the median predicted mean.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidate points (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidate points (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    score : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    t0 = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    n = len(mu)
    
    if n == 0:
        return np.array([])
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, np.median(mu))
    
    # Check time budget
    if time.time() - t0 > DEADLINE:
        # Fallback: simple UCB
        return -mu + sigma
    
    # Quantile-based normalization
    # Rank mu: lower mu is better for exploitation
    mu_ranks = np.argsort(np.argsort(mu)).astype(np.float64) / max(n - 1, 1)
    # mu_ranks in [0, 1], 0 = best (lowest mu), 1 = worst (highest mu)
    
    # Rank sigma: higher sigma is better for exploration
    sigma_ranks = np.argsort(np.argsort(sigma)).astype(np.float64) / max(n - 1, 1)
    # sigma_ranks in [0, 1], 0 = lowest uncertainty, 1 = highest uncertainty
    
    # For exploitation: we want low mu, so exploit_rank = 1 - mu_ranks (higher is better)
    exploit_rank = 1.0 - mu_ranks
    
    # For exploration: we want high sigma, so explore_rank = sigma_ranks (higher is better)
    explore_rank = sigma_ranks
    
    # Dynamic weight: shift toward exploration when f_best is much larger than median mu
    med_mu = np.median(mu)
    gap = f_best - med_mu
    
    # Normalize the gap relative to the scale of mu
    mu_range = np.ptp(mu)
    if mu_range < 1e-10:
        mu_range = 1e-10
    
    # If f_best is significantly above the median prediction, explore more
    # gap/mu_range > 0 means f_best is above median -> explore
    # gap/mu_range < 0 means f_best is below median -> exploit
    normalized_gap = gap / mu_range
    
    # Weight for exploration: sigmoid-like function of normalized_gap
    # When normalized_gap is large positive, w_explore -> 0.5 (balanced)
    # When normalized_gap is large negative, w_explore -> 0 (exploit)
    # When normalized_gap is positive but small, w_explore -> moderate
    w_explore = 0.5 / (1.0 + np.exp(-2.0 * normalized_gap))
    
    # Clamp to reasonable range
    w_explore = np.clip(w_explore, 0.1, 0.6)
    w_exploit = 1.0 - w_explore
    
    # Compute score as convex combination
    score = w_exploit * exploit_rank + w_explore * explore_rank
    
    # Add a small EI-like component for additional discrimination
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # But we use a simplified version based on the normalized values
    
    # Compute z-score: how many std devs f_best is above mu
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # EI (for minimization, positive when mu < f_best)
    # Standard EI formula
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Normalize EI to [0, 1] using quantiles
    if np.ptp(ei) > 1e-10:
        ei_norm = (ei - np.min(ei)) / (np.max(ei) - np.min(ei))
    else:
        ei_norm = np.zeros_like(ei)
    
    # Blend: 70% quantile-based, 30% EI
    score = 0.7 * score + 0.3 * ei_norm
    
    # Final safety check
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check time
    if time.time() - t0 > DEADLINE - 5:
        pass  # Return what we have
    
    return score
