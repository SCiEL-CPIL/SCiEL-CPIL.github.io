import numpy as np
import time

def heuristics(demands, capacity):
    """
    Heuristic matrix for ACO bin packing: attractiveness of placing item j after item i.
    Uses a product-to-waste ratio with a smooth tightness bonus (soft-min surrogate
    on residual capacity) and a mild size-balance term.
    """
    start_time = time.time()

    d = np.asarray(demands, dtype=np.float64)
    n = len(d)

    if n == 0:
        return np.empty((0, 0))

    # Pairwise sums and products
    sums = d[:, None] + d[None, :]
    products = d[:, None] * d[None, :]

    # Feasibility mask
    fits = sums <= capacity

    # Waste = capacity - sum (non-negative where fits)
    waste = capacity - sums

    # Safe denominator: for fitting pairs, waste + 1 >= 1; for non-fitting, use 1.0
    safe_denom = np.where(fits, waste + 1.0, 1.0)

    # Base score: product / (waste + 1) -- higher for tight, large-item pairs
    base_scores = products / safe_denom

    # Smooth tightness bonus: soft-min of (waste, capacity) via log-sum-exp.
    # For fitting pairs waste <= capacity, so soft-min ≈ waste (smaller waste = tighter = better).
    # We use a negative soft-min to reward tightness: -softmin(waste, capacity).
    # softmin(a,b) = -logaddexp(-a, -b)  (smooth min)
    # We only care about fitting pairs; for non-fitting we'll mask later.
    # Clip waste to [0, capacity] for numerical safety
    waste_clipped = np.clip(waste, 0.0, capacity)
    # softmin(waste_clipped, capacity) = -logaddexp(-waste_clipped, -capacity)
    soft_min_waste = -np.logaddexp(-waste_clipped, -capacity)
    # Tightness bonus: we want SMALL waste to be GOOD, so bonus = -soft_min (more negative waste → larger bonus)
    # But we want a positive contribution that decreases with waste.
    # Let's use: tight_bonus = capacity - soft_min_waste  (ranges from 0 to capacity, larger when waste is small)
    tight_bonus = capacity - soft_min_waste

    # Size-balance term: prefer pairs where items are complementary in size.
    # A simple measure: 1 / (1 + |d_i - d_j| / capacity) -- higher when sizes are similar? 
    # Actually for bin packing, complementary sizes (one large, one small) can be good.
    # Let's use: balance = 1 - |d_i - d_j| / capacity  (higher when sizes are similar)
    # But actually, for tight packing, we want items that fill space well.
    # Let's try: balance = min(d_i, d_j) / max(d_i, d_j) -- ratio, higher when similar
    # Or simply: 1 / (1 + |d_i - d_j|) -- mild penalty for size mismatch
    diff = np.abs(d[:, None] - d[None, :])
    balance = 1.0 / (1.0 + diff / capacity)

    # Combine: base score scaled by tightness bonus and balance
    # Normalize tight_bonus to [0, 1] range for blending
    tight_norm = tight_bonus / capacity  # in [0, 1]
    
    # Final score: base * (1 + alpha * tight_norm) * (0.5 + 0.5 * balance)
    # This keeps the base structure but adds smooth refinements
    alpha = 0.5  # weight for tightness bonus
    H_raw = base_scores * (1.0 + alpha * tight_norm) * (0.5 + 0.5 * balance)

    # Mask non-fitting pairs to 1e-9
    H = np.where(fits, H_raw, 1e-9)

    # Sanitize: replace any non-finite values
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H), H, 1e-9)

    # Shape check
    if H.shape != (n, n):
        raise ValueError("Wrong shape")

    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 290:
        return H

    return H
