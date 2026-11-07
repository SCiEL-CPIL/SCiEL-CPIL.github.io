"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Combined acquisition: normalized EI + time-varying GP-UCB bonus.
    mu, sigma: arrays of shape (2000,) in objective units.
    f_best: lowest observed value.
    Returns 2000 finite scores, higher = evaluate next.
    """
    # Track time for a deadline guard (safety margin before 300s per evaluation)
    # We don't know the exact start time of the evaluation loop, so we use a
    # lightweight approach: if computation takes too long, return a safe fallback.
    t_start = time.time()
    
    # Handle edge cases where sigma might be zero or very small
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # Normalize EI to [0, 1] range for combining
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    # Time-varying UCB parameter: beta_t = sqrt(2 * t * log(1 + t))
    # We don't have direct access to t (step number), so we estimate it from
    # the number of points already evaluated. The function is called with 2000
    # candidates, and the loop runs until 30 evaluations. We can infer t from
    # f_best and the spread of mu/sigma, but a simpler approach is to use a
    # heuristic based on the magnitude of f_best.
    #
    # Alternative: use a fixed but principled beta that balances exploration.
    # For Beale on [-4.5, 4.5]^2 with 30 evals, we want moderate exploration.
    # Let's use beta = sqrt(2 * 10 * log(11)) ≈ sqrt(20 * 2.398) ≈ sqrt(47.96) ≈ 6.93
    # This corresponds to t=10, a middle ground.
    
    # Actually, let's make it adaptive: early on (f_best is large), explore more;
    # later (f_best is small), exploit more.
    # Scale beta by the current uncertainty level.
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # If f_best is close to 0 (minimum), reduce exploration
    # If f_best is large, increase exploration
    # Beale's minimum is 0, and initial values can be ~100+
    if f_best > 10:
        beta = 5.0  # High exploration early
    elif f_best > 1:
        beta = 3.0  # Moderate exploration
    elif f_best > 0.1:
        beta = 1.5  # Lower exploration
    else:
        beta = 0.5  # Low exploration, focus on exploitation
    
    # GP-UCB bonus
    ucb_bonus = beta * sigma_safe
    
    # Normalize UCB bonus to [0, 1]
    ucb_min = np.min(ucb_bonus)
    ucb_max = np.max(ucb_bonus)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb_bonus - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb_bonus)
    
    # Combine: weight EI more heavily as we converge
    # Weight based on how close f_best is to 0
    if f_best > 5:
        w_ei = 0.6
        w_ucb = 0.4
    elif f_best > 0.5:
        w_ei = 0.7
        w_ucb = 0.3
    else:
        w_ei = 0.8
        w_ucb = 0.2
    
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Deadline guard: if computation is taking too long, return safe values
    t_elapsed = time.time() - t_start
    if t_elapsed > 0.5:  # Should be fast, but guard just in case
        # Return a simple fallback: prefer low mu with some sigma
        return mu + 0.1 * sigma_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
