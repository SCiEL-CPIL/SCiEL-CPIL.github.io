import numpy as np
import time

# Global deadline tracking
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Uses posterior sampling: draws a sample from N(mu, sigma) for each
    candidate and scores by the negative sample value (lower f is better,
    so higher score = lower sampled value). Cost-awareness is applied by
    scaling the exploration (sigma) inversely with cost, so that expensive
    regions get slightly less uncertainty-driven exploration.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviations.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Evaluation costs for each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're too close to the 330s limit, return a safe
    # fallback (pure exploitation via mu) to avoid timeout.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        mu_arr = np.asarray(mu)
        fallback_scores = -mu_arr
        return np.where(np.isfinite(fallback_scores), fallback_scores, 0.0)
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety clamps
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Cost-aware exploration scaling:
    # cost is in [1, 2]. We want expensive points to have slightly less
    # exploration. Use a mild inverse scaling: effective_sigma = sigma / (1 + alpha*(cost-1))
    # alpha controls how strongly cost reduces exploration.
    alpha = 0.3
    effective_sigma = sigma_safe / (1.0 + alpha * (cost_safe - 1.0))
    
    # Thompson Sampling: draw one sample from the posterior for each candidate
    # scores = -sample, so higher score means lower sampled objective value
    rng = np.random.default_rng()
    samples = mu + effective_sigma * rng.standard_normal(len(mu))
    
    # Negate because we want to minimize f, but acquisition should be maximized
    scores = -samples
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
