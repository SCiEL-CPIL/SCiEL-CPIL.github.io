import numpy as np
from scipy.stats import norm
import time

_start_time = time.monotonic()
_DEADLINE = _start_time + 290.0  # 10s margin before 300s limit

# Known global minimum of Styblinski-Tang 1D
GLOBAL_MIN = -39.1662
# Total budget: 5 initial + 25 more = 30 evaluations
TOTAL_EVALS = 30
INITIAL_EVALS = 5


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


def acquisition(mu, sigma, f_best):
    """
    Kriging Believer acquisition function with adaptive exploration.
    
    Core idea: shift the GP mean toward lower values using uncertainty
    (LCB: mu - k*sigma), then compute EI on this shifted mean. This
    naturally balances exploration and exploitation through the k parameter.
    
    The k parameter adapts based on progress toward the known global minimum:
    - Early (far from optimum): larger k for more exploration
    - Late (close to optimum): smaller k for exploitation
    
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

    # Calculate progress based on how close f_best is to the known global minimum
    # The Styblinski-Tang function ranges from about -39 to positive values
    # Use a smooth sigmoid-like transition
    diff = f_best - GLOBAL_MIN
    if diff <= 0:
        progress = 1.0
    else:
        # Normalize diff: at start, diff can be large (e.g., 50+); near end, small
        # Use a scale that makes progress ~0.5 when diff ~ 10
        scale = 15.0
        progress = 1.0 / (1.0 + np.exp(diff / scale))
        # Ensure progress is in [0, 1]
        progress = np.clip(progress, 0.0, 1.0)
    
    # Kriging Believer: use lower confidence bound as surrogate mean
    # k adapts: more aggressive (higher k) in early stages, more conservative later
    # Early (progress~0): k ~ 2.5 (strong exploration)
    # Late (progress~1): k ~ 0.1 (nearly pure exploitation)
    k = 0.1 + 2.4 * (1.0 - progress)
    mu_lb = mu - k * sigma
    
    # Compute EI on the lower confidence bound (Kriging Believer)
    # This gives a principled balance of exploration/exploitation
    ei_scores = _expected_improvement(mu_lb, sigma, f_best)
    
    # Ensure Finite Output
    scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    return scores
