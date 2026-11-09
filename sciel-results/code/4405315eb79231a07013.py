"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Calculate total prize and total weight for each item across all constraints
    total_prize = np.sum(prizes, axis=0)
    total_weight = np.sum(weights, axis=0)
    
    # Add small epsilon to avoid division by zero
    eps = 1e-9
    
    # Calculate global efficiency: total prize / total weight
    heur = total_prize / (total_weight + eps)
    
    # Ensure values are finite and positive
    # If total_weight is 0 (impossible with uniform [0,1] unless all 0, but even then eps handles it), heur is 0 or inf.
    # With eps, denominator is > 0.
    # If total_prize is 0, heur is 0.
    heur[~np.isfinite(heur)] = 1e-9
    heur[heur <= 0] = 1e-9
    
    return heur
# EVOLVE-BLOCK-END
