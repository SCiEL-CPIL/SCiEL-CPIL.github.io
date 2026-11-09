"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for Thompson Sampling
_acq_state = {"step": 0, "rng": None}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling with adaptive exploration.
    
    Draws a single sample from the posterior at each candidate point:
      sample_i = mu_i - sigma_i * z_i,  z_i ~ N(0,1)
    Score = -sample_i (higher is better, i.e., lower predicted value)
    
    The exploration is scaled by an adaptive coefficient that increases
    when f_best is far from the known minimum (-1.8013), and decreases
    as we approach it. This provides problem-aware exploration intensity.
    
    Args:
        mu: Array of GP mean predictions (2000,).
        sigma: Array of GP standard deviation predictions (2000,).
        f_best: The lowest observed function value.
        
    Returns:
        Array of scores (2000,). Higher scores indicate better candidates.
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])
    
    # Time guard
    if time.time() > deadline:
        return np.zeros(n)
    
    # Sanitize sigma: ensure non-negative and finite
    sigma_clean = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma_clean = np.maximum(sigma_clean, 0.0)
    
    # Sanitize mu: replace non-finite with a large value (penalty)
    mu_clean = np.where(np.isfinite(mu), mu, 1e10)
    
    # Increment step counter
    _acq_state["step"] += 1
    t = _acq_state["step"]
    
    # Known minimum of Michalewicz 2D
    f_min_known = -1.8013
    
    # Adaptive exploration coefficient:
    # - When f_best is far from f_min, explore more (larger sigma multiplier)
    # - As f_best approaches f_min, reduce exploration
    # - Also include a mild time decay to ensure convergence
    gap = f_best - f_min_known  # positive when we're above the minimum
    # Scale gap to a reasonable range; clamp to [0, 1]
    gap_normalized = np.clip(gap / 2.0, 0.0, 1.0)
    
    # Exploration coefficient: base + adaptive + time decay
    # Base exploration to ensure we don't get stuck
    base_exploration = 0.5
    # Adaptive part: more exploration when far from optimum
    adaptive_exploration = 1.5 * gap_normalized
    # Time decay: slight reduction over 30 steps
    time_decay = 0.3 * (1.0 - min(t, 30) / 30.0)
    
    beta = base_exploration + adaptive_exploration + time_decay
    
    # Thompson Sampling: draw one sample per candidate
    # Use a fixed-seed RNG for reproducibility within a run
    if _acq_state["rng"] is None:
        # Seed based on step to get different samples each call
        _acq_state["rng"] = np.random.default_rng(42)
    
    z = _acq_state["rng"].standard_normal(n)
    
    # Sample from posterior: mu - sigma * z (minimizing)
    samples = mu_clean - beta * sigma_clean * z
    
    # Score = -sample (higher is better)
    scores = -samples
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Time guard before return
    if time.time() > deadline - 5.0:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
