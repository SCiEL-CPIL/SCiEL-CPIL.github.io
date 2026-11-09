"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Add small epsilon to weights to avoid division by zero
    eps = 1e-6
    safe_weights = weights + eps
    
    # Calculate per-constraint efficiency ratios: prize / weight
    ratios = prizes / safe_weights
    
    # Clip ratios to a reasonable range to avoid extreme values in log/exp
    # This prevents overflow/underflow issues and handles zero prizes/weights
    ratios = np.clip(ratios, 1e-9, 1e9)
    
    # Calculate geometric mean of ratios across constraints (axis 0)
    # Geometric mean = exp(mean(log(x)))
    # This approach rewards items that are efficient across all dimensions,
    # penalizing any single dimension where the item is inefficient.
    log_ratios = np.log(ratios)
    mean_log_ratios = np.mean(log_ratios, axis=0)
    heur = np.exp(mean_log_ratios)
    
    # Ensure values are finite and positive
    heur[~np.isfinite(heur)] = 1e-9
    heur[heur <= 0] = 1e-9
    
    return heur
# EVOLVE-BLOCK-END
