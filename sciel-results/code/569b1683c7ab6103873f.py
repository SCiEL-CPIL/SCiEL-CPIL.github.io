"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    start_time = time.time()
    deadline = 300.0 - 5.0  # Safety margin

    # Convert to numpy arrays for consistent behavior
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Handle potential 2D prizes by reducing to 1D if necessary
    if prizes.ndim == 2:
        prize_vector = np.sum(prizes, axis=0)
    else:
        prize_vector = prizes.flatten()

    # Ensure weights is (5, 100)
    weight_sums = np.sum(weights, axis=0)
    weight_maxs = np.max(weights, axis=0)
    
    # Compute mean weight per item (average across constraints)
    weight_means = np.mean(weights, axis=0)
    
    # Compute standard deviation of weights per item
    weight_stds = np.std(weights, axis=0)
    
    # The baseline uses prize / (sum(weights) + eps) and scores ~1.0
    # Let's try a more nuanced cost function that accounts for both total bulk and concentration
    # cost = alpha * sum + beta * max + gamma * std
    # Since sum and max are correlated, let's try a combination
    
    # Idea: Penalize items that are heavy in multiple dimensions more than items heavy in one
    # But also penalize very high total weight
    # 
    # Let's try: cost = weight_sums + 0.5 * weight_maxs
    # This adds a moderate penalty for bottleneck-heavy items on top of total weight
    
    cost = weight_sums + 0.5 * weight_maxs + 1e-9
    heuristic = prize_vector / cost
    
    # Apply a moderate power to sharpen the distribution
    # Power > 1 amplifies high ratios relative to low ones
    heuristic = heuristic ** 1.3
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Check time
    if time.time() > deadline:
        return heuristic

    return heuristic

# EVOLVE-BLOCK-END
