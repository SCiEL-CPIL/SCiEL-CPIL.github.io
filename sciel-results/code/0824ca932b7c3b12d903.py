"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    # Favor tight fits: higher heuristic when pair sum is close to capacity
    # but not exceeding it. Use a smooth bell curve centered at capacity.
    # Also penalize pairs that are too small (waste space) or too large (don't fit).
    
    # For fitting pairs (sum <= capacity), reward closeness to capacity
    # For non-fitting pairs, use minimum value
    fill_ratio = pair / capacity  # 0 to ~1.33 (max pair is 99+99=198, 198/150=1.32)
    
    # Smooth reward: peaks at fill_ratio=1.0 (perfect fit), drops off for smaller
    # Use a Gaussian-like bump: exp(-alpha * (fill_ratio - 1)^2) for fitting pairs
    # But we want it to be 0 (or 1e-9) for non-fitting pairs
    
    # For fitting pairs: reward based on how close to capacity
    # Use: 1 + beta * (fill_ratio - 0.5)^2 for fitting, which rewards larger sums
    # Actually, let's use a different approach: reward = fill_ratio^k for fitting pairs
    # Higher k makes it more sensitive to filling the bin
    
    # Let's use: for fitting pairs, heuristic = (fill_ratio)^2 + 0.1
    # This rewards pairs that fill more of the bin, with a small base to avoid zeros
    
    fit_mask = pair <= capacity
    heuristic = np.where(fit_mask, (fill_ratio ** 2) + 0.1, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
