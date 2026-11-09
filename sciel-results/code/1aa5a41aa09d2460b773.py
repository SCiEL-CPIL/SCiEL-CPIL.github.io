import numpy as np
import time

# Module-level deadline guard (150s wall-clock limit, use 140s safety margin)
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for placing `item`. Higher score = better placement.
    
    Combines:
    1. Best-fit base (minimize remaining capacity)
    2. Sigmoid-based "next item" bonus (from inspiration 1)
    3. Smooth sigmoid fragmentation transitions (from inspiration 2/3)
    
    The key structural idea: instead of a fixed Gaussian penalty for the "bad zone",
    we use smooth sigmoid transitions to create a cost landscape that:
    - Linearly penalizes small remaining (best-fit)
    - Adds extra penalty in the fragmentation zone (T1 to T2)
    - Reduces penalty for large remaining (bins that can still accept many items)
    - Adds a sigmoid bonus for bins likely to fit the next item
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return best-fit as fallback
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full_like(remaining, -np.inf, dtype=float)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Remaining capacity in each bin
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid = remaining >= 0
    scores = np.full(len(bins), -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores
    
    rem_valid = remaining[valid]
    
    # Parameters for fragmentation-aware scoring
    T1 = 5.0    # below this, it's "good enough" (nearly full)
    T2 = 25.0   # above this, remaining space is still useful
    alpha = 0.6  # reduced weight for large remaining spaces
    
    # Smooth transitions using tanh
    eps = 1.0
    
    # Transition 1: from linear to penalized zone
    # At rem < T1, weight ~ 0; at rem > T1+eps, weight ~ 1
    trans1 = 0.5 * (1.0 + np.tanh((rem_valid - T1) / eps))
    
    # Transition 2: from penalized to reduced zone
    # At rem < T2-eps, weight ~ 0; at rem > T2, weight ~ 1
    trans2 = 0.5 * (1.0 + np.tanh((rem_valid - T2) / eps))
    
    # Fragmentation penalty: Gaussian bump centered in the bad zone
    center = (T1 + T2) / 2
    width = (T2 - T1) / 4
    frag_penalty = np.exp(-((rem_valid - center) ** 2) / (2 * width ** 2))
    
    # Combined cost with smooth transitions
    # Base cost: rem_valid
    # In fragmentation zone, add extra penalty
    # For large remaining, reduce the penalty
    cost = rem_valid * (1.0 + 2.0 * trans1 * frag_penalty) * (1.0 - (1.0 - alpha) * trans2)
    
    # Next-item bonus: sigmoid that rewards bins likely to fit the next item
    # Use current item as proxy for next item size
    next_item_est = item
    beta = 0.5 * item  # scaled bonus
    
    # Sigmoid: probability that bin can fit next item
    sigmoid = 1.0 / (1.0 + np.exp(-(rem_valid - next_item_est) / max(item, 1.0)))
    
    # Subtract cost, add bonus
    valid_scores = -cost + beta * sigmoid
    
    scores[valid] = valid_scores
    
    return scores
