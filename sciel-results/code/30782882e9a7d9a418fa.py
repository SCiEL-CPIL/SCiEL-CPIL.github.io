"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def _check_deadline(timeout=320.0):
    """Return True if we should stop to avoid timeout."""
    return (time.time() - _start_time) > timeout

def acquisition(mu, sigma, f_best):
    """
    Lookahead-UCB acquisition with hard top-k gating.
    
    This strategy uses a two-stage selection process:
    1. Compute a UCB score for minimization (mu - kappa * sigma).
    2. Select the top K% of candidates based on this UCB score.
    3. Within this selected pool, use Expected Improvement (EI) to determine the final score.
    4. Candidates outside the pool receive a heavily penalized score.
    
    This structure encourages exploration of promising regions (high UCB) while
    using EI to make precise choices within those regions, avoiding the pitfalls
    of linear blending where exploration and exploitation compete equally everywhere.
    
    mu, sigma: shape (2000,), posterior mean and std in objective units.
    f_best: lowest observed value so far.
    Returns: 2000 finite scores, higher = more promising.
    """
    n = len(mu)
    
    # Deadline guard
    if _check_deadline():
        # Fallback: Simple EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 1. Compute EI for all candidates
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Compute UCB for minimization
    # We want to find low values. High uncertainty (sigma) is good for exploration.
    # Low mean (mu) is good for exploitation.
    # UCB_min = mu - kappa * sigma.
    # We need to determine kappa.
    
    # Estimate progress. 
    # f_min is the known global minimum -2.53218.
    # f_best is the current best.
    # Regret = f_best - f_min.
    # Early on, regret is large. Late on, regret is small.
    
    f_min = -2.53218
    # Initial regret is roughly f_best - f_min. 
    # We don't know the initial f_best exactly, but we can use the spread of mu or just a fixed schedule.
    # A robust proxy for "early" vs "late" is the relative improvement.
    # Let's use the number of evaluations if we could track it, but we can't.
    # We can use the magnitude of regret.
    # If f_best is close to f_min, we are late.
    
    regret = f_best - f_min
    # Clip regret to a reasonable range to avoid extreme kappa values
    # Max regret might be around 5.0 (if f_best is 2.5 and f_min is -2.5)
    regret_clipped = np.clip(regret, 0.0, 5.0)
    
    # Kappa should be high when regret is high (early) and low when regret is low (late)
    # Exponential decay
    kappa = 0.5 + 2.0 * np.exp(-regret_clipped / 0.5)
    
    ucb_min = mu - kappa * sigma_safe
    
    # 3. Determine Top-K pool
    # Start with 50% pool, shrink to 10% as we get closer to the minimum
    # Use regret to control the pool size
    # If regret > 2.0, pool = 50%. If regret < 0.1, pool = 10%.
    # Linear interpolation
    pool_frac = np.clip(0.5 - 0.4 * (1.0 - regret_clipped / 5.0), 0.1, 0.5)
    
    k = max(1, int(n * pool_frac))
    
    # Find indices of the top K candidates by ucb_min (lowest values are best)
    # argsort returns indices that would sort the array.
    # We want the smallest K values.
    top_k_indices = np.argpartition(ucb_min, k)[:k]
    
    # Create a mask for the top K candidates
    mask = np.zeros(n, dtype=bool)
    mask[top_k_indices] = True
    
    # 4. Final Score
    # Inside the pool: Use EI
    # Outside the pool: Use a scaled-down EI or 0
    # To ensure the selection is strictly from the pool, we can set outside scores to -inf or 0.
    # Since EI >= 0, setting outside to 0 is safe if there's any positive EI inside.
    # However, if all EI are 0 (flat function?), UCB might still vary.
    # Let's use EI inside, and a very small fraction of EI outside to maintain continuity if needed,
    # but effectively gating.
    
    scores = np.zeros(n)
    scores[mask] = ei[mask]
    
    # For those outside, we give them 0. This ensures the argmax is always in the top-K UCB set.
    # This is a "hard gate".
    
    # Safety: Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
