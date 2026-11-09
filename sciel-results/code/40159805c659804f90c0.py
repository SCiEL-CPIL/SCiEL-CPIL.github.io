"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: 330s total, leave 30s margin
_DEADLINE = 300.0
_start_time = time.time()

def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition:
    1. Sample a single function value from the GP posterior at each candidate point.
       Since mu and sigma are provided as the posterior mean and std at the 2000 candidates,
       we can sample: f_sample = mu + sigma * z, where z ~ N(0, 1).
    2. The "score" for each candidate is -f_sample (since we want to minimize f).
       The evaluator will pick the candidate with the highest score, which corresponds
       to the lowest sampled f value.
    
    This is a structural change from EI/UCB: it uses stochastic sampling rather than
    deterministic scoring, which avoids hyperparameter tuning and naturally explores
    regions with high uncertainty (high sigma) when the sampled value happens to be low.
    """
    if _check_deadline():
        # Fallback: return EI-like scores to be safe
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (f_best - mu) / (sigma + 1e-12)
            from scipy.stats import norm
            scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Thompson Sampling: draw one sample from the posterior at each candidate
    # f_i ~ N(mu_i, sigma_i^2)
    # We sample z ~ N(0, 1) for each candidate
    z = np.random.standard_normal(mu.shape)
    f_samples = mu + sigma * z
    
    # We want to minimize f, so the "score" for the acquisition function should be
    # higher for better (lower f) candidates.
    # score = -f_samples
    scores = -f_samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
