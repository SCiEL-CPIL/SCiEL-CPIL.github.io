import numpy as np

def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_constraints = weights.shape[0]
    n_items = weights.shape[1]
    
    # Item prize: sum across rows
    item_prizes = np.sum(prizes, axis=0)
    
    capacity = 25.0
    
    # Total weight per constraint
    total_w = np.sum(weights, axis=1)
    
    # Bindingness: how binding each constraint is
    # utilization = total_w / capacity
    utilization = total_w / capacity
    # Higher utilization = more binding
    # Use exponential to emphasize
    binding = np.exp(2.0 * utilization)  # or utilization^k
    binding_sum = np.sum(binding)
    if binding_sum > 0:
        binding = binding / binding_sum
    
    # Per-constraint ratio
    safe_w = np.where(weights > 1e-9, weights, 1e-9)
    ratios = item_prizes[np.newaxis, :] / safe_w  # (n_constraints, n_items)
    
    # Weighted geometric mean of ratios (emphasizes binding constraints)
    # log ratio, weighted average, exp
    log_ratios = np.log(np.where(ratios > 1e-9, ratios, 1e-9))
    weighted_log = np.sum(binding[:, np.newaxis] * log_ratios, axis=0)
    geo_mean = np.exp(weighted_log)
    
    # Also compute unweighted harmonic mean (bottleneck)
    inv_ratios = 1.0 / np.where(ratios > 1e-9, ratios, 1e-9)
    harm_mean = float(n_constraints) / np.sum(inv_ratios, axis=0)
    
    # Min ratio (strictest bottleneck)
    min_ratio = np.min(ratios, axis=0)
    
    # Blend
    def norm_pos(a):
        a = np.where(a > 0, a, 1e-9)
        m = np.mean(a)
        if m > 0:
            a = a / m
        return a
    
    h = 0.4 * norm_pos(geo_mean) + 0.3 * norm_pos(harm_mean) + 0.3 * norm_pos(min_ratio)
    h = np.where(h > 0, h, 1e-9)
    
    return h
