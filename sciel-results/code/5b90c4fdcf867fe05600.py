"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidates.
    
    Uses a Thompson Sampling (Posterior Sampling) approach with an
    exploration-exploitation balance. This is a structural change from
    the previous EI+quantile-UCB hybrid to a sampling-based method
    that naturally balances exploration and exploitation.
    
    For minimization, we sample from the posterior at each candidate:
    y_i ~ N(mu_i, sigma_i^2)
    
    Then we score each candidate by how much it could potentially
    improve upon f_best, with a bias toward points that have high
    uncertainty (sigma) to encourage exploration.
    
    The score is: -y_sampled + beta * sigma
    where beta controls exploration. We adaptively set beta based on
    the ratio of current best to typical uncertainty.
    """
    t_start = time.time()
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    n = len(mu)
    
    # Deadline guard check
    if time.time() - t_start > 250:
        # Fall back to simple LCB
        return -mu + 1.0 * sigma
    
    # Adaptive exploration parameter
    # When f_best is close to 0 (minimum), we want more exploitation
    # When f_best is far from 0, we want more exploration
    # sigma_mean gives us a sense of the typical uncertainty scale
    sigma_mean = np.mean(sigma)
    if sigma_mean < 1e-8:
        sigma_mean = 1e-8
    
    # Adaptive beta: larger when we're still far from optimum
    # f_best is the lowest observed value; for Beale, min is 0
    # If f_best is large relative to sigma, we're in exploration phase
    exploration_ratio = f_best / (sigma_mean + 1e-8)
    # Clip to reasonable range
    exploration_ratio = np.clip(exploration_ratio, 0.1, 10.0)
    
    # Use a moderate exploration parameter
    # When exploration_ratio is high (far from optimum), explore more
    beta = 0.5 + 0.5 * np.log1p(exploration_ratio)
    beta = np.clip(beta, 0.3, 3.0)
    
    # Thompson Sampling: draw one sample from the posterior at each point
    # y_i ~ N(mu_i, sigma_i^2)
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(n)
    
    # Score: we want to maximize the chance of finding a new minimum
    # Primary signal: -sampled_value (lower sampled value = better)
    # Secondary signal: sigma (higher uncertainty = more potential)
    
    # Expected Improvement style: how much better than f_best could this be?
    # But using the sample: if sample < f_best, it's an improvement
    # Score = max(0, f_best - sample) + beta * sigma * indicator
    
    # Actually, let's use a cleaner formulation:
    # Score = -sample + beta * sigma
    # This rewards low sampled values AND high uncertainty
    
    scores = -samples + beta * sigma
    
    # Add a small EI-like term to bias toward points that are likely improvements
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Normalize EI to [0, 1] for blending
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max > ei_min + 1e-12:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    # Normalize scores to [0, 1] for blending
    s_min = np.min(scores)
    s_max = np.max(scores)
    if s_max > s_min + 1e-12:
        scores_norm = (scores - s_min) / (s_max - s_min)
    else:
        scores_norm = np.zeros_like(scores)
    
    # Blend: primarily use the Thompson sampling score, with a boost from EI
    w_ts = 0.7
    w_ei = 0.3
    final_scores = w_ts * scores_norm + w_ei * ei_norm
    
    # Scale back to a reasonable range
    if s_max > s_min + 1e-12:
        final_scores = final_scores * (s_max - s_min) + s_min
    
    # Ensure all scores are finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    # Deadline guard
    if time.time() - t_start > 250:
        return -mu + 1.0 * sigma
    
    return final_scores
# EVOLVE-BLOCK-END
