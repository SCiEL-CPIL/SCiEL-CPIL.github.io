"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling-inspired scoring.
    Samples objective values from the GP posterior and scores candidates
    by how likely they are to beat f_best, with a bonus for uncertainty.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (objective units, lower is better).
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (objective units).
    f_best : float
        Best observed function value so far (lower is better).

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    t0 = time.time()
    deadline = 325.0  # Safety margin before 330s limit

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)

    # Thompson Sampling: sample from posterior for each candidate
    # Use a deterministic pseudo-random seed based on candidate index for reproducibility
    # This creates a stochastic but deterministic exploration pattern
    n = len(mu)
    
    # Create a deterministic "random" offset for each candidate using a hash-like approach
    # This simulates Thompson Sampling without actual randomness, ensuring reproducibility
    # Use a simple LCG-based approach for speed
    rng = np.random.RandomState(42)
    samples = rng.standard_normal(n)
    
    # Sampled values from GP posterior
    sampled_values = mu + sigma_safe * samples
    
    # Score: probability that the sampled value is below f_best, 
    # plus a small uncertainty bonus
    # P(sampled < f_best) can be approximated by the CDF of the normal distribution
    # But we want a score that rewards low sampled values and high uncertainty
    
    # Primary score: how much better than f_best is the sampled value
    # Use a softmax-like transformation for smooth ranking
    improvement = f_best - sampled_values
    improvement = np.clip(improvement, -100, 100)
    
    # Normalize improvement to [0, 1] using sigmoid
    # Temperature controls the sharpness of the preference
    temp = max(abs(f_best), 1.0) * 0.1 + 0.01
    scores_primary = 1.0 / (1.0 + np.exp(-improvement / temp))
    
    # Uncertainty bonus: reward candidates with high sigma
    # This encourages exploration in uncertain regions
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        uncertainty_bonus = sigma_safe / sigma_max
    else:
        uncertainty_bonus = np.zeros_like(sigma_safe)
    
    # Blend primary score with uncertainty bonus
    # Weight: 85% primary (exploit promising samples), 15% uncertainty (explore)
    w_primary = 0.85
    w_uncertainty = 1.0 - w_primary
    scores = w_primary * scores_primary + w_uncertainty * uncertainty_bonus
    
    # Add a small penalty for very high mean values to avoid wasting evaluations
    # in obviously poor regions
    scale = max(abs(f_best), 1.0)
    mu_penalty = np.maximum(0.0, (mu - f_best) / (scale + 1e-12))
    mu_penalty = np.minimum(mu_penalty, 2.0)
    scores = scores - 0.03 * mu_penalty
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
