import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    valid = remaining >= 0
    
    if not np.any(valid):
        return scores
        
    r = remaining[valid]
    
    # Strategy: Prefer remaining space that's "useful" for future items
    # For Weibull items (many small, few large), target remaining ~ 20-30% of capacity
    # This leaves room for small items while not wasting too much space
    
    # Score: negative of |r - target|, with target being a "sweet spot"
    # Target: items that leave space for ~2-3 more small items
    # Weibull with typical params: mean ~ 8-15 for capacity 100
    # So target remaining ~ 20-40
    
    target = 25.0  # Target remaining space
    width = 30.0   # Width of preference
    
    # Parabolic score: higher when r is close to target
    scores[valid] = -(r - target) ** 2 / (2 * width ** 2)
    
    # Add a small penalty for very tight fits (r < 0.5 * item) to avoid fragmentation
    tight_penalty = np.where(r < 0.5 * item, 0.1 * (0.5 * item - r) / item, 0.0)
    scores[valid] -= tight_penalty
    
    return scores
