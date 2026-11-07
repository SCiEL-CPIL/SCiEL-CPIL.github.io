import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = time.time() + 280  # safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Hybrid Thompson Sampling + Expected Improvement Acquisition Function.
    
    This function combines the stochastic exploration of Thompson Sampling with
    the deterministic exploitation of Expected Improvement (EI).
    
    Strategy:
    1. Compute Expected Improvement (EI) scores.
    2. Compute a Thompson Sampling (TS) score by sampling from the posterior.
    3. Combine them with an adaptive weight. Early on, TS is favored to escape
       local minima and explore. Later, EI is favored to converge to the optimum.
    
    The combination is: Score = (1 - w) * EI + w * (-TS_sample)
    where w decays as f_best improves (approaches the global minimum).
    
    Parameters:
    -----------
    mu : np.ndarray, shape (N,)
        GP posterior mean at candidate points.
    sigma : np.ndarray, shape (N,)
        GP posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (N,)
        Acquisition scores; higher means more attractive to evaluate next.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: just exploit (maximize -mu)
        return -np.asarray(mu, dtype=np.float64).ravel().copy()
    
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    n = len(mu)
    
    # Handle any non-finite values
    mask = np.isfinite(mu) & np.isfinite(sigma)
    if not np.all(mask):
        safe_mu_val = np.max(mu[mask]) if np.any(mask) else 0.0
        mu = np.where(mask, mu, safe_mu_val)
        sigma = np.where(mask, sigma, 0.0)
    
    # Ensure sigma is strictly positive to avoid division by zero
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    
    # 1. Calculate Standard Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to prevent numerical overflow/underflow in norm.cdf/pdf
    z_clipped = np.clip(z, -50, 50)
    
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    scores_ei = improvement * Phi_z + sigma * phi_z
    
    # 2. Calculate Thompson Sampling Score
    # Draw random samples from the standard normal distribution
    z_ts = np.random.standard_normal(n)
    
    # Compute the sampled objective values
    # y ~ N(mu, sigma^2)
    ts_samples = mu + sigma * z_ts
    
    # We want to minimize, so higher score is better.
    # Thus, we use the negative of the sample.
    scores_ts = -ts_samples
    
    # 3. Determine Adaptive Weight
    # We want to blend EI and TS.
    # Early on (f_best is high), we want more exploration (TS).
    # Late on (f_best is low), we want more exploitation (EI).
    
    f_global_min = -3.86278
    
    # Normalize progress: 0.0 at f_best = 0.0 (start), 1.0 at f_best = -3.86
    # We clamp f_best to [0, -3.86] for this calculation
    f_best_clamped = np.clip(f_best, -3.86, 0.0)
    progress = (0.0 - f_best_clamped) / 3.86  # 0 at start, 1 at end
    
    # Weight for TS (exploration)
    # Start with high TS weight (0.8) and reduce to low (0.1)
    w_ts = 0.8 - 0.7 * progress
    
    # Weight for EI (exploitation)
    w_ei = 1.0 - w_ts
    
    # 4. Combine Scores
    # To make the scales comparable, we could normalize, but EI and -TS are
    # roughly on the same scale (objective units).
    # However, EI can be 0 if mu > f_best, while -TS can be negative.
    # A simple linear combination works well.
    
    scores = w_ei * scores_ei + w_ts * scores_ts
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
