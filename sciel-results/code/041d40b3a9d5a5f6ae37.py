import numpy as np

def heuristics(prizes, weights):
    # Per-constraint efficiency: prize / weight for each constraint
    # Avoid division by zero
    safe_weights = np.where(weights < 1e-9, 1e-9, weights)
    efficiency = prizes / safe_weights  # shape (5, 100)
    
    # Normalize weights per constraint to capture relative tightness
    # This makes the bottleneck ratio more meaningful across constraints
    weight_sums = weights.sum(axis=1, keepdims=True)  # shape (5, 1)
    norm_weights = weights / (weight_sums + 1e-9)  # shape (5, 100)
    
    # Harmonic mean of per-constraint efficiencies:
    #   n / sum(1/efficiency_i)
    # This penalizes items where ANY constraint has a low ratio more heavily
    # than arithmetic or geometric mean, acting as a smooth bottleneck measure.
    safe_eff = np.where(efficiency < 1e-9, 1e-9, efficiency)
    inv_eff = 1.0 / safe_eff  # shape (5, 100)
    harmonic_mean = 5.0 / np.sum(inv_eff, axis=0)  # shape (100,)
    
    # Total prize per item (reward high-value items)
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # Combine: primary signal is harmonic mean efficiency, secondary is total prize
    # Normalize both to [0, 1] range for balanced combination
    harm_norm = (harmonic_mean - harmonic_mean.min()) / (harmonic_mean.max() - harmonic_mean.min() + 1e-9)
    prize_norm = (total_prize - total_prize.min()) / (total_prize.max() - total_prize.min() + 1e-9)
    
    # Heuristic: emphasize harmonic mean efficiency, add prize as tiebreaker
    heuristic = harm_norm + 0.1 * prize_norm
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
