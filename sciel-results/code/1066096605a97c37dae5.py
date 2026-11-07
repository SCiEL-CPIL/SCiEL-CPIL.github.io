"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Computes attractiveness scores using a multi-constraint efficiency heuristic
    that blends the simple sum-ratio with a geometric-mean ratio to balance
    scale dominance and signal dilution.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Handle potential shape mismatches:
    # Expected (5, 100) i.e., (Constraints, Items)
    # If passed as (100, 5) i.e., (Items, Constraints), transpose.
    if prizes.ndim == 2:
        if prizes.shape == (100, 5):
            prizes = prizes.T
            weights = weights.T
        elif prizes.shape != (5, 100):
            # If shape is unexpected but close, try to reshape or handle gracefully.
            # However, based on the task, it should be 5x100 or 100x5.
            # If it's neither, we might have a bug, but let's not crash immediately.
            # We will assume the second dimension is items if the first is not 5.
            if prizes.shape[0] != 5 and prizes.shape[1] == 5:
                 prizes = prizes.T
                 weights = weights.T
            elif prizes.shape[0] == 5:
                 pass # Already correct
            else:
                 # Fallback: assume last dim is items? Or just proceed and let numpy errors handle it?
                 # To be safe, if we can't determine, we might return a dummy heuristic.
                 # But let's assume the standard two cases cover it.
                 pass
    else:
        # If 1D, it's ambiguous. But task says 2D.
        # If it's flat, we can't easily know which axis is which without more info.
        # We'll assume the input is valid 2D as per contract.
        pass

    # Ensure we are working with (5, 100)
    # If shape is still wrong, we might get errors later. 
    # Let's enforce the shape for the calculations below.
    if prizes.shape != (5, 100):
         # If we didn't fix it, try to reshape if size matches
         if prizes.size == 500:
             prizes = prizes.reshape(5, 100)
             weights = weights.reshape(5, 100)
         else:
             # If we can't fix it, return a neutral heuristic to avoid crashing
             # though this will likely score low.
             return np.ones(100)

    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # Total weight per item (sum across constraints)
    total_weight = weights.sum(axis=0)  # shape (100,)
    
    # Simple ratio: total_prize / total_weight
    # This is the baseline heuristic
    simple_ratio = total_prize / (total_weight + 1e-9)  # shape (100,)
    
    # Geometric mean ratio: (product of per-constraint ratios)^(1/5)
    # This penalizes items that are inefficient in any single constraint
    per_constraint_ratio = prizes / (weights + 1e-9)  # shape (5, 100)
    geo_ratio = np.exp(np.mean(np.log(per_constraint_ratio + 1e-9), axis=0))  # shape (100,)
    
    # Normalize both to [0, 1] range to avoid scale dominance
    def normalize(x):
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min < 1e-12:
            return np.ones_like(x)
        return (x - x_min) / (x_max - x_min)
    
    simple_norm = normalize(simple_ratio)
    geo_norm = normalize(geo_ratio)
    
    # Blend: equal weight for both signals
    blended = 0.5 * simple_norm + 0.5 * geo_norm
    
    # Ensure positive values (the evaluator replaces non-positive with 1e-9)
    blended = np.maximum(blended, 1e-9)
    
    return blended
# EVOLVE-BLOCK-END
