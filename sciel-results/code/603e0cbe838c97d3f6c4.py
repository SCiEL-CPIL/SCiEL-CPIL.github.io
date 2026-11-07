import numpy as np


def heuristics(prizes, weights):
    """
    prizes: (5, 100) array, uniform in [0, 1]
    weights: (5, 100) array, uniform in [0, 1]
    Returns: (100,) array of heuristic scores, larger = more attractive
    """
    # Total prize per item
    item_prizes = prizes.sum(axis=0)  # (100,)
    
    # Normalize weights by capacity (25 for each constraint)
    capacity = 25.0
    normalized_weights = weights / capacity  # (5, 100)
    
    # Penalty: 1 + sum of squared normalized weights per item
    # This penalizes items that consume a large fraction of any constraint's capacity
    weight_penalty = 1.0 + (normalized_weights ** 2).sum(axis=0)  # (100,)
    
    # Heuristic: prize / penalty
    heuristic = item_prizes / weight_penalty
    
    # Ensure positive values (though prizes are in [0,1], so item_prizes >= 0)
    # The ACO will clamp non-positive to 1e-9, but let's be safe
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
