import numpy as np

def heuristics(prizes, weights):
    """
    Returns a heuristic array of shape (100,) for the MCKP.
    Uses geometric mean of prize-to-weight ratios, with a scarcity premium
    for the most binding constraint per item.
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes)
    weights = np.asarray(weights)
    
    # Clip values to prevent division by zero and log of zero
    eps = 1e-9
    p = np.clip(prizes, eps, None)
    w = np.clip(weights, eps, None)
    
    # Per-constraint ratios: shape (5, 100)
    ratios = p / w
    
    # Geometric mean of ratios
    # log(geo_mean) = mean(log(ratios))
    log_ratios = np.log(ratios)
    mean_log_ratio = np.mean(log_ratios, axis=0)
    
    # Convert back to linear scale
    geo_mean = np.exp(mean_log_ratio)
    
    # Identify the most binding constraint for each item
    # The constraint where weight is largest fraction of total weight across constraints
    total_w = np.sum(w, axis=0)  # shape (100,)
    total_w = np.maximum(total_w, eps)
    frac = w / total_w  # shape (5, 100), fraction of each constraint's weight
    
    # Most binding constraint: argmax over constraints of fraction
    most_binding = np.argmax(frac, axis=0)  # shape (100,)
    
    # Get the ratio for the most binding constraint
    items_idx = np.arange(100)
    binding_ratios = ratios[most_binding, items_idx]  # shape (100,)
    
    # Capacity is 25 for all constraints
    capacity = 25.0
    
    # Remaining capacity fraction for the most binding constraint
    # If weight in that constraint is w[i], remaining fraction is (capacity - w[i]) / capacity
    # Scarcity premium: inverse of remaining fraction, so tighter constraints get higher premium
    binding_w = w[most_binding, items_idx]  # shape (100,)
    remaining_frac = (capacity - binding_w) / capacity
    remaining_frac = np.clip(remaining_frac, eps, 1.0)
    scarcity_premium = 1.0 / remaining_frac
    
    # Blend: use geometric mean as base, multiply by a tempered scarcity premium
    # Use a power less than 1 to avoid over-amplification
    tempered_premium = np.power(scarcity_premium, 0.5)
    
    heuristic = geo_mean * tempered_premium
    
    # Ensure all values are positive finite
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
