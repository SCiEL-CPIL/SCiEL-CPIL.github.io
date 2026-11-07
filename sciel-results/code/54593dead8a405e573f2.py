"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for ACO of multi-constraint knapsack.
    
    prizes: (5, 100) or (100,) uniform [0,1]
    weights: (5, 100) uniform [0,1]
    Returns: (100,) array, larger = more attractive.
    """
    start_time = time.time()
    
    # Ensure correct shapes and types
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_constraints = weights.shape[0]
    n_items = weights.shape[1]
    
    # Item prize: sum across constraint dimensions if 2D, else the value
    if prizes.shape[0] == 1:
        item_prizes = prizes[0].copy()
    else:
        item_prizes = np.sum(prizes, axis=0)
    
    # Capacity per constraint
    capacity = 25.0
    
    # Total weight per constraint
    total_w = np.sum(weights, axis=1)  # (n_constraints,)
    
    # Bindingness: how close each constraint is to being fully utilized
    # A constraint with total_w close to capacity is more binding
    # utilization = total_w / capacity; bindingness = utilization^2 (quadratic emphasis)
    utilization = total_w / capacity
    bindingness = np.clip(utilization, 0, 2) ** 2
    # Normalize bindingness to sum to 1 for weighted combination
    binding_sum = np.sum(bindingness)
    if binding_sum > 0:
        binding_weights = bindingness / binding_sum
    else:
        binding_weights = np.ones(n_constraints) / n_constraints
    
    # Per-constraint efficiency: prize / weight_c
    # Use safe division
    safe_w = np.where(weights > 1e-9, weights, 1.0)  # if weight is ~0, treat as 1 (no penalty)
    eff = item_prizes[np.newaxis, :] / safe_w  # (n_constraints, n_items)
    
    # Harmonic mean of efficiencies (captures weakest link)
    # HM = n / sum(1/eff_c)
    inv_eff = 1.0 / np.where(eff > 1e-9, eff, 1e-9)
    hm = float(n_constraints) / np.sum(inv_eff, axis=0)  # (n_items,)
    
    # Geometric mean of efficiencies (alternative multi-constraint aggregation)
    log_eff = np.log(np.where(eff > 1e-9, eff, 1e-9))
    gm = np.exp(np.mean(log_eff, axis=0))  # (n_items,)
    
    # Min-ratio: the minimum per-constraint efficiency (strictest weakest link)
    min_eff = np.min(eff, axis=0)  # (n_items,)
    
    # Weighted harmonic mean: weight each constraint's inverse efficiency by bindingness
    weighted_inv = np.sum(binding_weights[:, np.newaxis] * inv_eff, axis=0)
    weighted_hm = 1.0 / weighted_inv  # (n_items,)
    
    # Blend: primarily harmonic mean (weakest link), with geometric mean and min-ratio
    # The harmonic mean already emphasizes the weakest constraint
    # Blend: 0.5 * hm + 0.3 * gm + 0.2 * min_eff
    # But normalize each component first to have mean 1
    
    def normalize_positive(arr):
        arr = np.where(arr > 0, arr, 1e-9)
        m = np.mean(arr)
        if m > 0:
            arr = arr / m
        return arr
    
    hm_n = normalize_positive(hm)
    gm_n = normalize_positive(gm)
    min_n = normalize_positive(min_eff)
    whm_n = normalize_positive(weighted_hm)
    
    # Final blend
    # Harmonic mean is the core (captures multi-constraint bottleneck)
    # Geometric mean smooths out extreme values
    # Weighted HM emphasizes binding constraints
    h = 0.35 * hm_n + 0.25 * gm_n + 0.20 * whm_n + 0.20 * min_n
    
    # Ensure positive
    h = np.where(h > 0, h, 1e-9)
    
    # Time check - if we've used too much time, return immediately
    elapsed = time.time() - start_time
    if elapsed > 290:
        return h
    
    # Additional refinement: rank-based adjustment
    # Items in the top quartile by prize should get a small boost
    # Items in the bottom quartile by total weight should get a small boost
    prize_rank = np.argsort(np.argsort(item_prizes)) / max(n_items - 1, 1)
    total_item_w = np.sum(weights, axis=0)
    weight_rank = np.argsort(np.argsort(total_item_w)) / max(n_items - 1, 1)  # lower weight = higher rank
    
    # Small multiplicative boost for high-prize, low-weight items
    boost = 1.0 + 0.1 * prize_rank + 0.05 * (1.0 - weight_rank)
    h = h * boost
    
    # Final normalization to mean 1
    mean_h = np.mean(h)
    if mean_h > 0:
        h = h / mean_h
    
    return h
# EVOLVE-BLOCK-END
