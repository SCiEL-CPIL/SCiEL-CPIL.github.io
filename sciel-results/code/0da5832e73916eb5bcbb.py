"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function for cost-aware Bayesian optimization.
    
    Uses stochastic sampling from the posterior predictive distribution,
    normalized by evaluation cost, to naturally balance exploration and exploitation.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units.
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value.
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x) where x in [0,1]^6).
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard
    t_start = time.time()
    DEADLINE_MARGIN = 5.0
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid issues
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: draw one sample from the posterior predictive
    # For minimization, we want to find low values, so we sample from N(mu, sigma)
    # and use the negative as the score (lower sampled value = better)
    # But we also need to normalize by cost
    
    # Draw a single sample from the GP posterior for each candidate
    rng = np.random.default_rng()
    eps = rng.standard_normal(size=mu.shape)
    samples = mu + sigma * eps
    
    # For minimization, lower samples are better
    # We want higher scores for lower samples, so use negative
    # Normalize by cost: score = -sample / cost
    # This naturally balances: low function value AND low cost are both rewarded
    
    # Use a cost-aware normalization: divide by cost^alpha where alpha tunes cost sensitivity
    # alpha = 1 is standard cost-normalized Thompson Sampling
    alpha = 1.0
    
    scores = -samples / np.power(cost, alpha)
    
    # Add a small deterministic component to break ties and provide stability
    # Use the GP mean as a secondary signal
    # This helps when the stochastic sample is noisy
    mu_norm = (mu - np.min(mu)) / (np.max(mu) - np.min(mu) + 1e-9)
    cost_norm = (cost - np.min(cost)) / (np.max(cost) - np.min(cost) + 1e-9)
    
    # Blend: primarily Thompson Sampling, with a small deterministic component
    # The deterministic component rewards low predicted mean and low cost
    det_score = -(mu_norm) * (1.0 - 0.5 * cost_norm)
    
    # Normalize the Thompson Sampling scores to be comparable
    ts_min = np.min(scores)
    ts_max = np.max(scores)
    if ts_max > ts_min + 1e-9:
        ts_norm = (scores - ts_min) / (ts_max - ts_min)
    else:
        ts_norm = np.zeros_like(scores)
    
    # Blend: 80% Thompson Sampling, 20% deterministic
    w_ts = 0.8
    w_det = 0.2
    
    final_scores = w_ts * ts_norm + w_det * (1.0 - det_score)
    
    # Ensure all scores are finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    # Deadline check
    elapsed = time.time() - t_start
    if elapsed > 300.0 - DEADLINE_MARGIN:
        return final_scores
    
    return final_scores
# EVOLVE-BLOCK-END
