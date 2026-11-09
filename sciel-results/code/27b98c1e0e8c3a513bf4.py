import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for each item in a multiple-constraint knapsack.
    
    Args:
        prizes: array of shape (5, 100) - prize values for each constraint/item
        weights: array of shape (5, 100) - weight costs for each constraint/item
        
    Returns:
        array of shape (100,) - attractiveness score per item (larger = more attractive)
    """
    start_time = time.time()
    # Safety margin: 320s
    deadline = 320.0
    
    try:
        prizes = np.asarray(prizes, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        
        # Ensure shapes are as expected (5, 100)
        if prizes.shape != (5, 100) or weights.shape != (5, 100):
            # If shapes are wrong, return a neutral heuristic
            n_items = prizes.shape[1] if prizes.ndim > 1 else 100
            return np.ones(n_items) * 1e-9

        # Total prize per item (sum across the 5 dimensions)
        values = prizes.sum(axis=0)
        
        # The bottleneck constraint for each item is the one with the maximum weight.
        # An item is "expensive" if it consumes a lot of any single resource.
        # Using max weight in the denominator penalizes items that are heavy in
        # any specific constraint, which is critical for multi-constraint feasibility.
        max_weights = weights.max(axis=0)
        
        # Heuristic score: Value Density relative to the tightest constraint
        # Add epsilon to avoid division by zero
        epsilon = 1e-9
        scores = values / (max_weights + epsilon)
        
        # Handle non-finite values (e.g., if values were inf, though inputs are [0,1])
        scores = np.nan_to_num(scores, nan=1e-9, posinf=1e9, neginf=1e-9)
        
        # Task requirement: Heuristic entries that are not positive become 1e-9
        scores = np.where(scores > 0, scores, 1e-9)
        
        # Check deadline
        if time.time() > deadline:
            return np.ones(100) * 1e-9
            
        return scores

    except Exception:
        # If any error occurs, return a neutral heuristic to avoid invalid result
        return np.ones(100) * 1e-9
