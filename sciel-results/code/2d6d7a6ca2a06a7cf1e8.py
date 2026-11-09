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
    
    # Fragmentation-aware scoring: penalize bins where the item occupies
    # a moderate fraction (20-50%) of remaining capacity, which creates
    # small unusable fragments. Prefer near-zero residual (tight fit) or
    # large residual (preserves flexibility for future items).
    
    # Ratio of item to remaining capacity
    ratio = item / (rem + 1e-10)
    
    # Base score: prefer smaller remaining (best-fit direction)
    # Use a moderate non-linear transformation
    base_score = -(rem + 1e-6) ** 0.5
    
    # Fragmentation penalty: peak penalty when ratio is in [0.2, 0.5]
    # Use a smooth bump function centered at 0.35
    center = 0.35
    width = 0.15
    penalty = np.exp(-0.5 * ((ratio - center) / width) ** 2)
    
    # Scale the penalty
    penalty_scale = 1.0
    adjusted_score = base_score - penalty_scale * penalty
    
    scores[feasible] = adjusted_score
    
    if time.time() > deadline:
        return scores
        
    return scores
