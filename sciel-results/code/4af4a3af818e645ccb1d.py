import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Strategy: Thompson Sampling with adaptive exploration bonus
    # Thompson Sampling draws from the posterior N(mu, sigma^2) and picks the
    # sample with the lowest value (for minimization). This provides natural
    # exploration-exploitation balance: high sigma regions have wide sample
    # distributions, so they're more likely to produce very low samples.
    #
    # We add a small deterministic exploration bonus based on sigma to ensure
    # that even if the random draw is unlucky in a high-uncertainty region,
    # that region still gets some preference. This prevents the algorithm from
    # completely ignoring uncertain areas due to bad random luck.
    
    # Create a deterministic seed from the optimization state for reproducibility
    # within a single acquisition call, but varying across calls as f_best and mu change
    seed_raw = abs(f_best) * 1e6 + np.sum(np.abs(mu)) + np.sum(sigma)
    seed = int(seed_raw) % (2**32)
    
    rng = np.random.default_rng(seed)
    
    # Draw standard normal samples
    eps = rng.standard_normal(mu.shape[0])
    
    # Thompson Sampling: Draw from the posterior distribution N(mu, sigma^2)
    samples = mu + sigma * eps
    
    # For minimization, lower samples are better.
    # Score = -sample (higher score means we want to evaluate this point)
    ts_scores = -samples
    
    # Add a small exploration bonus: points with higher sigma get a small boost
    # This ensures that even if the random draw is bad, high-uncertainty regions
    # still have some chance of being selected
    # The bonus is scaled relative to the spread of the samples to remain balanced
    sample_range = np.max(samples) - np.min(samples)
    if sample_range < 1e-12:
        sample_range = 1e-12
    
    # Exploration bonus: proportional to sigma, normalized by sample spread
    exploration_bonus = 0.1 * sigma / sample_range
    
    # Combine Thompson Sampling score with exploration bonus
    scores = ts_scores + exploration_bonus
    
    # Check deadline
    if time.time() > deadline:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
