import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    t0 = time.time()
    deadline = t0 + 115.0
    
    remaining = bins - item
    feasible = remaining >= 0
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(feasible):
        return scores
    
    rem = remaining[feasible]
    
    # Fragmentation-aware scoring:
    # Penalize bins where the item takes a moderate fraction (20-50%) of remaining space,
    # which creates small unusable fragments. Favor bins where the item either
    # nearly fills the bin (high utilization) or leaves a large enough remainder.
    ratio = item / (rem + 1e-6)
    
    # Base score: prefer smaller remainders (best-fit like)
    base = -np.sqrt(rem + 1e-6)
    
    # Fragmentation penalty: peak penalty when ratio is in [0.2, 0.5]
    # Use a smooth bump function centered at ratio=0.35
    center = 0.35
    width = 0.15
    penalty = -5.0 * np.exp(-((ratio - center) ** 2) / (2 * width ** 2))
    
    # Bonus for very high utilization (ratio close to 1.0)
    high_util_bonus = 2.0 * np.exp(-((ratio - 0.95) ** 2) / (2 * 0.05 ** 2))
    
    scores[feasible] = base + penalty + high_util_bonus
    
    if time.time() > deadline:
        return scores
        
    return scores
