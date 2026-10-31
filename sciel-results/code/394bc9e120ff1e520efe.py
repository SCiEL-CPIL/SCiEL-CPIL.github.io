"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = preferred bin.
    
    Strategy: Adaptive Best-Fit with fragmentation awareness.
    - For large items (> 0.5 * capacity), strongly prefer Best Fit (minimize leftover).
    - For small items, use a blend that avoids excessive fragmentation by
      slightly preferring bins that are already 'used up' (have less free space
      relative to their original capacity, but since we don't track original capacity,
      we approximate by preferring bins with moderate remaining space to avoid
      creating many nearly-empty bins).
    
    The key insight from the knowledge base: a smooth sigmoid transition that
    favors Best Fit for large items and becomes neutral (not Worst Fit) for small items.
    
    We interpret "neutral" as: for small items, the score should not strongly favor
    either the tightest or loosest fit, but rather a moderate fit. We can achieve this
    by using a smooth function that transitions from Best Fit behavior to a more
    balanced approach.
    
    Let's define:
    - remaining = bins - item (for bins that can fit the item)
    - For bins that cannot fit the item, score = -inf
    
    For bins that can fit:
    - We want to minimize 'waste' but also avoid fragmentation.
    - A common approach: score = -remaining / (bins + epsilon)  [relative waste]
      But this might not capture the sigmoid behavior.
    
    Let's try: score = -remaining * sigmoid(item / capacity_ref)
    where capacity_ref is a reference capacity (100 in the task).
    
    Actually, let's think differently. The knowledge says:
    "favors Best Fit for large items and becomes neutral for small items"
    
    Best Fit = minimize remaining (so score = -remaining)
    Neutral = score should not depend much on remaining, or should prefer medium remaining.
    
    Let's use: score = -remaining * w(item) + bias(item)
    where w(item) transitions from 1 (large item) to 0 (small item) via sigmoid.
    And bias(item) could be 0 for simplicity, or a small constant.
    
    But "neutral" might mean: for small items, we don't care about remaining, so
    all bins get similar scores. We can achieve this by making the coefficient of
    -remaining approach 0 for small items.
    
    Let capacity = 100.0 (from task description)
    threshold = 0.5 * capacity = 50.0
    w = sigmoid((item - threshold) / (0.1 * capacity))  # smooth transition around 50
    
    For large items (item >> 50), w ≈ 1, so score ≈ -remaining (Best Fit)
    For small items (item << 50), w ≈ 0, so score ≈ 0 (neutral, all bins equal)
    
    But if all bins get score 0, the tie-breaking will be arbitrary. We might want
    a small tie-breaker. Let's add a tiny bias toward bins with more remaining capacity
    for small items to avoid fragmentation? Or toward less remaining?
    
    The knowledge says "becomes neutral (rather than Worst Fit)". So it should NOT
    prefer Worst Fit. Neutral likely means: no strong preference, so tie-breaking
    can be based on something else, like bin index or a very small random-like perturbation.
    But we can't use random. Let's use a very small negative coefficient for remaining
    when item is small, so it's still slightly Best Fit but much weaker.
    
    Let's try:
    w = sigmoid((item - 50.0) / 10.0)
    score = -remaining * (0.1 + 0.9 * w)
    
    For small items: score ≈ -remaining * 0.1 (weak Best Fit)
    For large items: score ≈ -remaining * 1.0 (strong Best Fit)
    
    This avoids the fragmentation penalty of pure Worst Fit while retaining adaptive structure.
    """
    capacity = 100.0
    threshold = 0.5 * capacity  # 50.0
    scale = 0.1 * capacity  # 10.0
    
    # Compute remaining capacity
    remaining = bins - item
    
    # Mask for bins that can fit the item
    can_fit = remaining >= 0
    
    # Sigmoid weight: 0 for small items, 1 for large items
    # Use a numerically stable sigmoid
    x = (item - threshold) / scale
    # Clip x to avoid overflow in exp
    x_clipped = np.clip(x, -50, 50)
    w = 1.0 / (1.0 + np.exp(-x_clipped))
    
    # Score: -remaining * (0.1 + 0.9 * w)
    # For bins that can't fit, score = -inf
    coeff = 0.1 + 0.9 * w
    scores = -remaining * coeff
    
    # Set infeasible bins to -inf
    scores = np.where(can_fit, scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
